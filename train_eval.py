# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam


# 权重初始化，默认xavier，这里可以先不着急看明白，知道有这么个事就行
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    #### 先做好准备工作 ####
    start_time = time.time()#先记录以下最开始的开始时间
    model.train()#训练模式开启
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')#验证集最佳损失先设置为无穷大
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    #### 开始训练吧 ####
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # 遍历获得一个一个minibatch
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)                 #前向传播一下
            model.zero_grad()                       #梯度清零一下
            loss = F.cross_entropy(outputs, labels) #计算交叉熵损失
            loss.backward()                         #反向传播一下
            optimizer.step()                        #优化一下
            if total_batch % 100 == 0:#每一百个minibatch就验证一下
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()                    #取当前batch的labels作为真值
                predic = torch.max(outputs.data, 1)[1].cpu()#取输出的最大值的类别作为预测值
                train_acc = metrics.accuracy_score(true, predic)#计算一下所有当前batch数据的训练准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)#计算一下对于所有验证集的验证集正确率和验证集损失
                if dev_loss < dev_best_loss:#如果验证损失小于最佳验证损失
                    dev_best_loss = dev_loss#更新最佳验证损失
                    torch.save(model.state_dict(), config.save_path)#保存当前模型
                    improve = '*'
                    last_improve = total_batch#保存当前batch数
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)#看一下到现在花了多少时间吧
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)#在测试集上测试一下数据吧

#测试数据函数
def test(config, model, test_iter):
    #
    model.load_state_dict(torch.load(config.save_path))#首先加载一下我们的最佳模型
    model.eval()#评估模式启动
    start_time = time.time()#记一个开始时间
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)#计算测试集正确率、测试集损失、
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

# 评估模型函数，需要参数：配置文件、模型对象、数据集的迭代器、是否是测试
def evaluate(config, model, data_iter, test=False):
    model.eval()#评估模式启动
    loss_total = 0#计算总损失用的
    predict_all = np.array([], dtype=int)#记录所有的预测值
    labels_all = np.array([], dtype=int)#记录所有的真实值
    with torch.no_grad():#不用计算梯度
        for texts, labels in data_iter:#遍历数据集的一个batch
            outputs = model(texts)#对当前batch进行计算
            loss = F.cross_entropy(outputs, labels)#计算当前batch的损失
            loss_total += loss#当前batch损失加到总损失中
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)