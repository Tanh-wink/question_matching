import paddle
import numpy as np

from paddlenlp.datasets import MapDataset


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query1"], example["query2"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def convert_example_pair(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query1"], example["query2"]

    encoded_inputs1 = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    encoded_inputs2 = tokenizer(
        text=title, text_pair=query, max_seq_len=max_seq_length)

    input_ids = encoded_inputs1["input_ids"]
    token_type_ids = encoded_inputs1["token_type_ids"]

    input_ids2 = encoded_inputs2["input_ids"]
    token_type_ids2 = encoded_inputs2["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, input_ids2, token_type_ids2, label
    else:
        return input_ids, token_type_ids, input_ids2, token_type_ids2

def convert_example_measure(example, tokenizer, max_seq_length=512, is_test=False):

    query1, query2 = example["query1"], example["query2"]

    input_ids1 = tokenizer(
        text=query1, max_seq_len=max_seq_length)["input_ids"]

    input_ids2 = tokenizer(
        text=query2, max_seq_len=max_seq_length)["input_ids"]


    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids1, input_ids2, label
    else:
        return input_ids1, input_ids2