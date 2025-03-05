# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)

#预训练模型档案映射，预训练模型对应的下载地址
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'#bert配置文件名
TF_WEIGHTS_NAME = 'model.ckpt'#权重参数文件名，tf格式的

############################以下是一些与模型构建相关的功能函数和类############################
#加载tf权重参数函数
def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ 在pytorch模型中加载tf checkpoints
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)#检查点路径
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # 从tf模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v 和 adam_m 是 AdamWeightDecayOptimizer 中用于计算 m 和 v 的变量，
        # 但对于使用预训练模型来说并不是必需的
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model
#gelu函数
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
#
def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
############################以上是一些与模型构建相关的功能函数和类############################

############################以下是transformer中的一些基础层############################
#embedding层
class BertEmbeddings(nn.Module):
    """从 词嵌入（word embeddings）、位置嵌入（position embeddings）和标记类型嵌入（token_type embeddings） 构造最终的嵌入表示。
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)#初始化embedding层，[语料库大小，隐藏层大小]，就意味着语料库中的每一个词都以隐藏层大小进行词嵌入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)#位置编码层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)#token类型嵌入层

        #self.LayerNorm 未使用蛇形命名（snake_case），以保持与 TensorFlow 模型变量名称一致，
        # 并确保能够加载任何 TensorFlow 检查点文件。
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)#layernorm层，这个暂时先不要关心，用就行了
        self.dropout = nn.Dropout(config.hidden_dropout_prob)#dropout层

    #input_ids[8, 128, 768]
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)#序列长度 128
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

####自注意力层，到这里，我们的网络每一层的注意力计算就完成了，给一个输入序列，输出进行自注意力计算后的输出序列
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads#定义多少头，比如12头
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)#每个头的size，比如64
        self.all_head_size = self.num_attention_heads * self.attention_head_size#所有头的size之和，比如768，又回到之前词向量的维度了
        #三个矩阵的构建
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

#### 跟在自注意力计算之后的线性变换，到这里我们就完成了完整的注意力计算，注意这里的线性变化是注意力计算里的，不是后边的前馈神经网络中的
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)#当然这里也要进行失活
        hidden_states = self.LayerNorm(hidden_states + input_tensor)#也要进行层归一化
        return hidden_states

#### 到这里我们就定义好了一个完整的注意力计算层，给一个输入序列，输出一个相同维度的输出序列，这个序列会直接再次输入给我们定义的注意力层
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)#自注意力计算
        self.output = BertSelfOutput(config)#自注意力计算后进行全连接层进行线性变换

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

####这个是最后一层前馈网络的一部分，将最后一层的输出序列映射为intermediate_size，这里是3072维，
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        #### 注意：BertIntermediate这个层的特殊之处就在于，这一层是带有激活函数的非线性变换，这个是属于前馈神经网络的一部分，请和注意力计算中的线性变换有所区分
        # 处理激活函数的选择：
        # - `config.hidden_act` 可能是字符串（例如 'gelu'、'relu'）。
        # - 如果 `hidden_act` 是字符串，则从 `ACT2FN` 字典中获取对应的激活函数。
        # - 在 Python 2 下，还需要兼容 `unicode` 类型的 `hidden_act`。
        # - 如果 `hidden_act` 不是字符串（可能是直接传入的自定义激活函数），则直接使用它。
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]# 从映射字典 `ACT2FN` 获取激活函数
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

####这个是最后一层前馈网络的一部分，将中间表示映射回hidden_size（768维），也就是词向量嵌入维度
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)

#定义bert中的每一层
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        #### intermediate+output其实就是一个前馈神经网络，BertAttention中的BertSelfOutput是注意力计算中的线性变换，请加以区分
        # 为什么最后一层做过了BertSelfOutput的前馈神经网络之后还要再做一次intermediate+output这个前馈神经网络呢？
        # 因为SelfOutput(X)=LayerNorm(Dropout(Dense(X))+X)，也就是BertSelfOutput只是做了线性变换，
        # 而FFN(X)=LayerNorm(Dropout(Dense2(GELU(Dense 1(X))))+X)，也就是BertIntermediate这一层做的是非线性变换，因为它有激活函数
        self.attention = BertAttention(config)#注意力层:BertSelfAttention+BertSelfOutput
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

#这里定义了encoder的结构
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)#根据配置文件创建一个BertLayer层的实例
        # 使用 nn.ModuleList 创建多个 BertLayer 层，并存入 self.layer
        # 这里使用了 copy.deepcopy(layer) 来确保每个层都是独立的副本，而不是共享同一个实例
        # `config.num_hidden_layers` 指定了 BERT 的隐藏层数量，例如 BERT-Base 有 12 层，BERT-Large 有 24 层
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 我们通过简单地取第一个标记（token）对应的隐藏状态来对模型进行“池化”（pooling）。
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
############################ 以上是transformer中的一些基础层 ############################

############################ 以下就是一些不同用途的bert模型 ############################
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

#一个抽象类，用于处理权重初始化，并提供下载和加载预训练模型的简单接口。
class BertPreTrainedModel(nn.Module):
    """ 一个抽象类，用于处理权重初始化，并提供下载和加载预训练模型的简单接口。 """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "参数 config 在 {} 中应为 BertConfig 类的一个实例。"
                "要从 Google 预训练模型创建模型，请使用 model = {}.from_pretrained(PRETRAINED_MODEL_NAME)。 ".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ 初始化权重.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 与 TensorFlow 版本略有不同，后者使用 truncated_normal 进行初始化
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        从预训练模型文件或 PyTorch 状态字典 (state_dict) 实例化一个 BertPreTrainedModel。
        如果需要，将下载并缓存预训练模型文件。

        Params:
            pretrained_model_name_or_path：可以是以下之一：
                - 一个字符串，表示要加载的预训练模型名称，可选项包括：
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - 一个路径或 URL，指向包含以下文件的预训练模型存档：
                    . `bert_config.json` 模型的配置文件
                    . `pytorch_model.bin` 一个 BertForPreTraining 实例的 PyTorch 权重文件
                - 一个路径或 URL，指向包含以下文件的预训练模型存档：
                    . `bert_config.json`模型的配置文件
                    . `model.chkpt` 一个 TensorFlow 权重检查点
            from_tf: 是否从本地保存的 TensorFlow 检查点加载权重。
            cache_dir: 一个可选路径，用于存储缓存的预训练模型文件夹。
            state_dict: 一个可选的状态字典 (collections.OrderedDict 对象)，可用于替代 Google 预训练模型的权重。
            *inputs, **kwargs: 用于特定 Bert 类的额外输入，例如：
                (ex: num_labels 用于 BertForSequenceClassification)
        """
        #### 首先我们需要加载预训练模型的参数，这里有点难，可以先跳过
        state_dict = kwargs.get('state_dict', None)# 从 kwargs 中获取 'state_dict' 参数，如果没有则设置为 None
        kwargs.pop('state_dict', None)# 从 kwargs 中移除 'state_dict' 参数，防止后续重复使用
        cache_dir = kwargs.get('cache_dir', None)# 从 kwargs 中获取 'cache_dir' 参数，如果没有则设置为 None
        kwargs.pop('cache_dir', None)# 从 kwargs 中移除 'cache_dir' 参数，防止后续重复使用
        from_tf = kwargs.get('from_tf', False)# 从 kwargs 中获取 'from_tf' 参数，如果没有则默认设置为 False
        kwargs.pop('from_tf', None)# 从 kwargs 中移除 'from_tf' 参数，防止后续重复使用

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:# 如果传入的 pretrained_model_name_or_path 在预训练模型地址映射中
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]# 从映射中获取相应的档案文件路径
        else:
            archive_file = pretrained_model_name_or_path# 否则，直接使用传入的路径作为档案文件路径
        # 如果需要，重定向到缓存
        try:
            # 尝试通过 cached_path 函数解析档案文件路径，并指定缓存目录 cache_dir
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            # 如果解析失败，则记录错误日志，提示模型名称未在模型列表中找到，
            # 并且假定 archive_file 是一个路径或 URL，但未能找到任何相关文件
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None# 返回 None 表示加载失败

        if resolved_archive_file == archive_file:# 判断解析后的档案文件路径是否与原始档案文件路径相同
            logger.info("loading archive file {}".format(archive_file))# 如果相同，直接记录加载档案文件的日志信息
        else:# 如果不同，说明是从缓存中加载，记录缓存路径的日志信息
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None# 初始化临时目录变量，默认设为 None
        if os.path.isdir(resolved_archive_file) or from_tf:# 如果解析后的档案文件路径是一个目录，或者模型是从 TensorFlow 加载的
            serialization_dir = resolved_archive_file# 直接将序列化目录设置为解析后的档案文件路径
        else:
            # 否则，从压缩包中提取内容到临时目录
            # 创建一个临时目录
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:# 使用 tarfile 模块打开压缩包，并将其内容全部提取到临时目录中
                archive.extractall(tempdir)
            serialization_dir = tempdir# 设置序列化目录为临时目录
        # 加载 config文件
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # 向后兼容旧的命名格式
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        #### 到这里我们已经拿到了预训练的参数，但是只有参数是没用的，我们还需要网络的实例现在来创建网络对象吧
        #从json中读取我们的config对象
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # 实例化模型
        model = cls(config, *inputs, **kwargs)
        #### 我们已经拿到了我们的网络实例了，现在我们要来加载我们的预训练参数了
        if state_dict is None and not from_tf:# 如果 state_dict 为 None 且不是来自 TensorFlow，则需要加载 PyTorch 权重
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)# 构建权重文件的完整路径，将序列化目录和权重文件名拼接
            state_dict = torch.load(weights_path, map_location='cpu')# 从指定路径加载权重到 state_dict 中，且加载到 CPU 内存中
        if tempdir:# 如果存在临时目录，则进行清理操作
            # 清理临时目录
            shutil.rmtree(tempdir)
        if from_tf:# 如果权重来自 TensorFlow 检查点
            # 直接从 TensorFlow 检查点加载权重
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            # 直接调用函数 load_tf_weights_in_bert，从 TensorFlow 检查点加载权重并返回
            return load_tf_weights_in_bert(model, weights_path)
        # 如果权重来自PyTorch则 从 PyTorch state_dict加载
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:# 如果键中包含 'gamma'，则将其替换为 'weight'
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:# 如果键中包含 'beta'，则将其替换为 'bias'
                new_key = key.replace('beta', 'bias')
            if new_key:# 如果新键不为空，说明发生了替换，则记录旧键和新键
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):# 根据记录的旧键和新键，更新 state_dict 中的键
            state_dict[new_key] = state_dict.pop(old_key)# 将新键的值设置为原旧键对应的值，并删除旧键

        # 初始化列表，用于存储加载过程中缺失的键、意外的键和错误消息
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # 复制 state_dict，以便 _load_from_state_dict 可以修改它
        metadata = getattr(state_dict, '_metadata', None)# 先获取 state_dict 中可能存在的 '_metadata' 属性
        state_dict = state_dict.copy()# 复制 state_dict
        if metadata is not None:# 如果存在元数据，则将其重新赋值到复制后的 state_dict 中
            state_dict._metadata = metadata

        def load(module, prefix=''):# 定义一个递归函数，用于将 state_dict 中的参数加载到模型的各个子模块中
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})# 根据前缀获取局部元数据，如果没有则使用空字典
            # 调用模块自带的 _load_from_state_dict 方法加载参数，同时记录缺失键、意外键和错误信息
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():# 对当前模块的每个子模块递归调用 load 函数，更新前缀
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''# 初始化加载的起始前缀为空字符串
        # 如果模型没有 bert 属性，但 state_dict 中存在以 'bert.' 开头的键，则说明需要加上前缀 'bert.'
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)# 从模型根模块开始加载 state_dict 中的参数
        if len(missing_keys) > 0:# 如果存在缺失的键，记录日志信息
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:# 如果加载过程中存在错误信息，则抛出运行时异常，并输出错误详情
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        #### 至此我们终于完成了创建网络的工作，我们实例化了一个网络，并且加载了预训练的参数，接下来返回这个模型，就可以进行训练了
        return model

#BertModel继承自BertPreTrainedModel，这个类用来实例化一个BertModel，暂时我们先只了解这个模型即可
# 后面可以根据不同的任务选择不同的模型
class BertModel(BertPreTrainedModel):
    """BERT 模型（"Bidirectional Embedding Representations from a Transformer"）
    参数:
        config: BertConfig 类的一个实例，包含用于构建新模型的配置。
    输入:
        `input_ids`: a torch.LongTensor 形状为 [batch_size, sequence_length]
            表示词汇表中的单词标记索引（请参考 extract_features.py、run_classifier.py 和 run_squad.py 中的标记预处理逻辑）。
        `token_type_ids`: （可选）torch.LongTensor，形状为 [batch_size, sequence_length]，标记类型索引，取值范围为 [0, 1]。
                            类型 0 对应 sentence A，类型 1 对应 sentence B（详见 BERT 论文）。
        `attention_mask`: （可选）torch.LongTensor，形状为 [batch_size, sequence_length]，取值范围为 [0, 1]。
                            用于处理输入序列长度小于当前批次最大序列长度的情况，通常用于注意力机制，以适应批次内不同长度的句子。
        `output_all_encoded_layers`: 布尔值，控制 encoded_layers 输出的内容（见下文）。默认为 True。
    输出: 返回一个包含 (encoded_layers, pooled_output) 的元组：
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - 当 output_all_encoded_layers=True 时，输出一个列表，包含每个注意力层末尾的完整编码隐藏状态序列（即 BERT-Base 12 层，BERT-Large 24 层）。
                    每个编码隐藏状态为 torch.FloatTensor，形状为 [batch_size, sequence_length, hidden_size]。
            - 当 output_all_encoded_layers=False 时，仅输出最后一个注意力层的完整隐藏状态序列，形状为 [batch_size, sequence_length, hidden_size]。
        `pooled_output`: torch.FloatTensor，形状为 [batch_size, hidden_size]。
            该值是通过分类器在隐藏状态的第一个标记 (CLS) 上进行训练得到的结果，适用于下一句预测（Next-Sentence Prediction） 任务（详见 BERT 论文）。
    使用案例:
    ```python
    # 已经转成WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    #### 首先我们来初始化bert模块的网络结构，组装embedding层、encoder模块和pooler层，这几个层在其他地方有所定义
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)#初始化embedding层
        self.encoder = BertEncoder(config)#初始化encoder
        self.pooler = BertPooler(config)#初始化pooler层，简单地取第一个标记（token）对应的隐藏状态进行非线性变换而已
        # 当然这个函数的实现还要看具体任务，当前情感分类任务就用第一个标记即可
        self.apply(self.init_bert_weights)#调用初始化权重函数，这个函数在父类定义好了，但是这个初始化后面可能被预训练模型参数覆盖哈
    #### 现在开始定义前向传播函数
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:#如果没有掩码，那就是全1掩码
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:#如果没有类型，那就是全0类型
            token_type_ids = torch.zeros_like(input_ids)

        # 我们从一个二维张量掩码创建一个三维注意力掩码。
        # 其尺寸为 [batch_size, 1, 1, to_seq_length]，
        # 这样可以广播为 [batch_size, num_heads, from_seq_length, to_seq_length]。
        # 这个注意力掩码比 OpenAI GPT 中用于因果注意力（causal attention）的三角形掩码更简单，
        # 这里只需准备好广播维度即可。
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 由于 attention_mask 在我们希望关注的位置取值为 1.0，在被屏蔽的位置取值为 0.0，
        # 这个操作将创建一个张量，使得我们希望关注的位置为 0.0，而被屏蔽的位置为 -10000.0。
        # 由于该张量会在 softmax 之前加到原始分数上，
        # 这实际上等效于完全移除这些被屏蔽的位置。
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)# 将输入id进行embedding
        print (np.array(embedding_output.data.cpu().numpy()).shape)#打印embedding之后的shape，debug用的，训练的时候要关闭哦，因为训练数据太多了，每个都打么
        encoded_layers = self.encoder(embedding_output,#对embedding后的向量列表进行encoder
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]#获取最后一层的输出序列
        print (np.array(sequence_output.data.cpu().numpy()).shape)
        pooled_output = self.pooler(sequence_output)#进行一下pool
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output#返回最后结果


class BertForPreTraining(BertPreTrainedModel):
    """带有预训练头部的 BERT 模型。
    该模块由 BERT 模型和两个预训练头部构成：
        - 掩码语言建模头部，
        - 下一句预测头部。

    参数：
        config：一个 BertConfig 类实例，用于构建新模型的配置。

    输入：
        `input_ids`: 形状为 [batch_size, sequence_length] 的 torch.LongTensor，
                    包含词汇表中单词的 token 索引（参见脚本 `extract_features.py`、`run_classifier.py` 和 `run_squad.py` 中的 token 预处理逻辑）。
        `token_type_ids`: 可选的形状为 [batch_size, sequence_length] 的 torch.LongTensor，
                            包含选自 [0, 1] 的 token 类型索引。类型 0 对应 “句子 A”，类型 1 对应 “句子 B”（详情请参阅 BERT 论文）。
        `attention_mask`: 可选的形状为 [batch_size, sequence_length] 的 torch.LongTensor，
                            其中的索引取自 [0, 1]。当当前批次中输入序列长度小于最大输入序列长度时，该掩码用于标识有效部分，通常用于处理批次中句子长度不一的情况。
        `masked_lm_labels`: 可选的掩码语言建模标签，形状为 [batch_size, sequence_length] 的 torch.LongTensor，
                            标签取值范围为 [-1, 0, ..., vocab_size]。所有值为 -1 的标签会被忽略（掩码处理），只对 [0, ..., vocab_size] 范围内的标签计算损失。
        `next_sentence_label`: 可选的下一句预测标签，形状为 [batch_size] 的 torch.LongTensor，
                            标签取值范围为 [0, 1]，其中 0 表示下一句为连续句子，1 表示下一句为随机句子。

    输出:
        如果 `masked_lm_labels` 和 `next_sentence_label` 均不为 `None`：:
            输出 total_loss，即掩码语言建模损失与下一句预测损失之和。
        如果 `masked_lm_labels` 或 `next_sentence_label` 为 `None`：
            输出一个元组，包含：
            - 掩码语言建模 logits，其形状为 [batch_size, sequence_length, vocab_size]，
            - 下一句预测 logits，其形状为 [batch_size, 2]。


    示例用法：
    ```python
    # 已经转换为 WordPiece token 索引
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """带有掩码语言模型头的BERT模型。
        该模块由BERT模型及其后接的掩码语言模型头组成。

    参数:
        config: 一个 BertConfig 类的实例，用于构建新模型的配置。

    输入:
        `input_ids`: 形状为 [batch_size, sequence_length] 的 torch.LongTensor，
                    其中包含词汇表中单词的标记索引（参见脚本 `extract_features.py`、`run_classifier.py` 和 `run_squad.py` 中的标记预处理逻辑）。
        `token_type_ids`: 一个可选的形状为 [batch_size, sequence_length] 的 torch.LongTensor，包含在 [0, 1] 中选择的标记类型索引。
                            类型 0 对应于 “句子 A”，类型 1 对应于 “句子 B”（更多细节请参见 BERT 论文）。
        `attention_mask`: 一个可选的形状为 [batch_size, sequence_length] 的 torch.LongTensor，其中的值为 [0, 1]。
                            当输入序列长度小于当前批次中的最大输入序列长度时，该 mask 用于注意力机制，通常用于处理批次中句子长度不一的情况。
        `masked_lm_labels`: 掩码语言建模标签，为形状为 [batch_size, sequence_length] 的 torch.LongTensor，
                            其中包含从 [-1, 0, ..., vocab_size] 中选择的索引。所有被设置为 -1 的标签将被忽略（即掩码），损失仅对 [0, ..., vocab_size] 范围内的标签进行计算。


    输出:
        如果 `masked_lm_labels` 不为 `None`：
            返回掩码语言建模的损失值。
        如果 `masked_lm_labels` 为 `None`：
            返回形状为 [batch_size, sequence_length, vocab_size] 的掩码语言建模 logits。


    示例用法:
    ```python
    # 已经转换为 WordPiece 标记 id
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
