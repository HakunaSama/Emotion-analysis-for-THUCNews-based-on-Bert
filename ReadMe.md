这是一个简易版本的pytorch框架的bert模型实例，参考的是google官方的tensorflow版本的bert源码，核心功能是完全一样的，但是更加精简便于阅读和理解bert的运行原理和逻辑，本实例的任务是新闻数据的文本分类任务
本项目结构
bert_pretrain

​    bert_condfig.json（bert模型配置文件）

​    pytorch_model.bin（）

​    vocab.txt（语料库文件）

models

​    bert.py（模型定义）

pytorch_pretrained

​    _init_.py

​    _main_.py

​    convert_gpt2_checkpoint_to_pytorch.py

​    convert_openai_checkpoint_to_pytorch.py

​    convert_tf_checkpoint_to_pytorch.py

​    convert_transfo_xl_checkpoint_to_pytorch.py

​    file_utils.py

​    modeling.py

​    modeling_gpt2.py

​    modeling_openai.py

​    modeling_transfo_xl.py

​    modeling_transfo_xl_utilities.py

​    optimization.py（优化器代码）

​    optimization_openai.py（）

​    tokenization.py（分词器）

​    tokenization_gpt2.py

​    tokenization_openai.py

​    tokenization_transfo_xl.py

THUCNews
run.py
train_eval.py
utils.py