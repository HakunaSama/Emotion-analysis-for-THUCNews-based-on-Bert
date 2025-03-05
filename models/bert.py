# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集路径
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集路径
        self.test_path = dataset + '/data/test.txt'                                  # 测试集路径
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果保存路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'                              # 预训练模型路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 预训练分词器
        self.hidden_size = 768                                          # 隐藏层大小

####定义模型，其实就是在BertModel的外面做了一些简单的包装
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)#创建模型中的bert模块，并且加载我们的预训练模型
        for param in self.bert.parameters():#所有的参数都不冻结都可以被训练
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)#最最后的全连接层，用来分类

    def forward(self, x):
        context = x[0]  # 输入的句子，
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)#用bert模块前向传播一下，进行encoder层计算
        out = self.fc(pooled)#输出进入到fc层
        return out#
