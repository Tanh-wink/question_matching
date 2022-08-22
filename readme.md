千言-问题匹配。任务旨在判断两个自然问句之间的语义是否相似

比赛官网：https://aistudio.baidu.com/aistudio/competition/detail/130/0/submit-result，报名参加即可下载数据集

原始代码github：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching/question_matching

快速开始
1. 主要运行环境

nvidia driver version 440.33.01

cuda 10.2

cudnn 7.6.0 

paddlepaddle-gpu==2.2.2

paddlenlp==2.2.5

如果要使用paddle分布式多卡训练，cuda10.2库需要libnccl.so，即安装nccl。

具体安装nccl教程请参考https://blog.csdn.net/Thanours/article/details/124078380?spm=1001.2014.3001.5501

2. 代码结构说明
以下是本项目主要代码结构及说明：

question_matching/
├── model.py # 匹配模型组网
├── data.py # 训练样本的数据读取、转换逻辑
├── predict.py # 模型预测脚本，输出测试集的预测结果: 0,1
└── train.py # 模型训练评估

数据准备
本项目使用竞赛提供的 LCQMC、BQ、OPPO 这 3 个数据集的训练集合集作为训练集，使用这 3 个数据集的验证集合集作为验证集。

运行如下命令生成本项目所使用的训练集和验证集，您在参赛过程中可以探索采取其它的训练集和验证集组合，不需要和基线方案完全一致。

模型训练  
运行如下命令，即可复现本项目中基于 ERNIE-Gram 的基线模型:

$unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0,1,2,3" train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./checkpoints \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0  
可支持配置的参数：

train_set: 训练集的文件。  
dev_set：验证集数据文件。  
rdrop_coef：可选，控制 R-Drop 策略正则化 KL-Loss 的系数；默认为 0.0, 即不使用 R-Drop 策略。  
train_batch_size：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。  
learning_rate：可选，Fine-tune的最大学习率；默认为5e-5。  
weight_decay：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。  
epochs: 训练轮次，默认为3。  
warmup_proption：可选，学习率 warmup 策略的比例，如果 0.1，则学习率会在前 10% 训练 step 的过程中从 0 慢慢增长到 learning_rate, 而后再缓慢衰减，默认为 0.0。  
init_from_ckpt：可选，模型参数路径，热启动模型训练；默认为None。  
seed：可选，随机种子，默认为1000。  
device: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。  
程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的save_dir中。  

训练过程中每一次在验证集上进行评估之后，程序会根据验证集的评估指标是否优于之前最优的模型指标来决定是否存储当前模型，如果优于之前最优的验证集指标则会存储当前模型，否则则不存储，因此训练过程结束之后，模型存储路径下 step 数最大的模型则对应验证集指标最高的模型, 一般我们选择验证集指标最高的模型进行预测。  

如：

checkpoints/  
├── model_10000  
│   ├── model_state.pdparams  
│   ├── tokenizer_config.json  
│   └── vocab.txt  
└── ...  
NOTE:  

如需恢复模型训练，则可以设置init_from_ckpt， 如init_from_ckpt=checkpoints/model_100/model_state.pdparams。  
开始预测  
训练完成后，在指定的 checkpoints 路径下会自动存储在验证集评估指标最高的模型，运行如下命令开始生成预测结果:  

$ unset CUDA_VISIBLE_DEVICES
python -u \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/model_10000/model_state.pdparams" \
    --batch_size 128 \
    --input_file "${test_set}" \
    --result_file "predict_result.csv"

