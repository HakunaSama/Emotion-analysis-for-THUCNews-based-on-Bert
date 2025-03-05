# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

#### step3 进入到我们的数据集构造函数中来
def build_dataset(config):

    #把data文件夹中的dev、train、test都读进来
    def load_dataset(path, pad_size=32):#padsize=32，一句话最长长度是32，多的截断，少的填充
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            # 遍历每一行
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')#获取当前行的文本content和标签label
                token = config.tokenizer.tokenize(content)#进行分词操作，这里就是直接分字，中文一般都是直接分字嗷
                token = [CLS] + token#在前面加上分类特殊字符，这个token最后的结果（768维度特征）放入全连接层然后进行分类预测
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)#把所有的字转成id

                if pad_size:
                    if len(token) < pad_size:#没有达到最大长度
                        #类似于[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))#做一个mask掩码
                        #类似于[101,704,1290,1957,2094,....0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        token_ids += ([0] * (pad_size - len(token)))#对应的typeid，这里只有一句话，是这句话的不变，填充的全部变成0
                    else:#达到最大长度
                        mask = [1] * pad_size#什么都不要遮盖
                        token_ids = token_ids[:pad_size]#全部使用原始值
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))#将处理好的token、对应的标签、文本长度、掩码存入contents
                # 额外说一句，其实这里的mask是没有必要的，
                # 因为token_ids里为0的就是不需要计算注意力的地方，但是为了保证结构一致能够适用于其他任务，这里还是要有掩码
        return contents
    train = load_dataset(config.train_path, config.pad_size)#获取训练数据集
    dev = load_dataset(config.dev_path, config.pad_size)#获取验证数据集
    test = load_dataset(config.test_path, config.pad_size)#获取测试数据集
    #### step3 好了，到这里我们就完全创建好了我们的数据集
    return train, dev, test

#创建数据集迭代器函数，通过这个迭代器，就能分批次获取数据集中的数据
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size#每一个批次的样本数
        self.batches = batches#总共有多少个batch
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):#返回batch数量
        if self.residue:#如果batch数量不是整数，则向上取整
            return self.n_batches + 1
        else:#如果batch数量是整数，直接返回
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
