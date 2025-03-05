# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
# --model bert

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # 模型名，我们就默认是bert就好
    x = import_module('models.' + model_name)# 动态导入bert,py模块
    #### step1根据数据集的名字来创建我们的配置文件，主要是配置一些路径和一些超参数
    config = x.Config(dataset)
    # 设置一些随机数种子，保证每次实验都可以复现，这个就不要关心了
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    #### step2 好了 首先我们来加载我们的数据集，我们的配置文件中保存有我们的数据集路径
    train_data, dev_data, test_data = build_dataset(config)#根据配置文件读取数据，点进去看
    #### step4 现在创建这些数据集的迭代器，用来给我们训练的时候使用
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    #### 创建模型，这个要点进去看具体实现哦
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
