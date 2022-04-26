from functools import partial
import argparse
import os
import random
import time
import logging

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from data import create_dataloader, read_text_pair, convert_example, convert_example_pair
from model import QuestionMatching, QuestionMatching_2stage

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = "/home/th/paddle/question_matching"
data_dir = project_dir + "/data"

def getArgs():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, default='/home/th/paddle/question_matching/data/train_merge.txt', required=False, help="The full path of train_set_file")
    parser.add_argument("--dev_set", type=str, default='/home/th/paddle/question_matching/data/dev.txt', required=False, help="The full path of dev_set_file")
    parser.add_argument("--save_dir", default=project_dir+'/checkpoint_gpu2', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_step", default=500, type=int, help="Step interval for evaluation.")
    parser.add_argument("--log_step", default=100, type=int, help="Step interval for logging and printing loss.")
    parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
    parser.add_argument("--warmup_proportion", default=1000, type=int, help="Linear warmup proption over the training process.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--union', choices=['logits', 'co_logits'], default="co_logits", help="Choice which logits to select answer.")
    parser.add_argument("--gpuids", type=str, default="1", required=False, help="set gpu ids which use to perform")
    parser.add_argument("--rdrop_coef", default=0.1, type=float, help="The coefficient of"
        "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")

    args = parser.parse_args()
    return args
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(args, model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    total_num = 0
    start = time.time()
    
    for batch in data_loader:
        input_ids, token_type_ids, input_ids2, token_type_ids2, labels = batch
        total_num += len(labels)
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids, 
            input_ids2=input_ids2,
            token_type_ids2=token_type_ids2
            )
        if args.rdrop_coef > 0 :
            logits, _, co_logits = outputs
            if args.union == "co_logits":
                logits = co_logits
        else:
            logits, _ = outputs
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    cost_times = time.time()-start
    logger.info("dev_loss: {:.4}, accuracy: {:.4}, cost_times:{:.4} s, eval_speed: {:.4} item/ms".format(
        np.mean(losses), accu, cost_times, (cost_times / total_num) * 1000))
    model.train()
    metric.reset()
    return accu


def train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)
    train_ds = load_dataset(
        read_text_pair, data_path=args.train_set, is_test=False, lazy=False)

    dev_ds = load_dataset(
        read_text_pair, data_path=args.dev_set, is_test=False, lazy=False)
    
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        'ernie-gram-zh')
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        'ernie-gram-zh')
    
    trans_func = partial(
        convert_example_pair,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input2
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment2
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    # model = QuestionMatching(pretrained_model, rdrop_coef=args.rdrop_coef)
    model = QuestionMatching_2stage(pretrained_model, rdrop_coef=args.rdrop_coef)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    logger.info(f"The number of examples in train set: {len(train_ds)}")
    logger.info(f"The number of examples in dev set: {len(dev_ds)}")
    logger.info(f"All training steps: {num_training_steps}")

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = nn.loss.CrossEntropyLoss()

    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        
        for step, batch in enumerate(train_data_loader, start=1):
            
            input_ids, token_type_ids, input_ids2, token_type_ids2,labels = batch
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, 
                            input_ids2=input_ids2, token_type_ids2=token_type_ids2)
            
            if args.rdrop_coef > 0 :
                logits, kl_loss, co_logits = outputs
                if args.union == "co_logits":
                    logits = co_logits
            else:
                logits, kl_loss = outputs
            correct = metric.compute(logits, labels)
            metric.update(correct)
            acc = metric.accumulate()

            ce_loss = criterion(logits, labels)
            if kl_loss > 0:
                loss = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss = ce_loss
            
            global_step += 1
            if global_step % args.log_step == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, train_acc: %.4f, speed: %.2f step/s"
                    % (global_step, epoch, loss, ce_loss, kl_loss, acc, args.log_step / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0 and epoch > 2:
                
                accuracy = evaluate(args, model, criterion, metric, dev_data_loader)
                if accuracy > best_accuracy:
                    save_dir = os.path.join(args.save_dir,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
                    logger.info(f"****({best_accuracy}------->{accuracy})****")
                    logger.info(f"Saved the best model at {save_param_path}")
                    best_accuracy = accuracy

            if global_step == args.max_steps:
                return

def run():
    args = getArgs()
    
    train(args)


if __name__ == "__main__":
    
    run()