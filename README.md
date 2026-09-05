Demo1 —— 基于 BERT 的中文新闻文本分类
一、项目结构
demo1/
│
├── data/
│   ├── train_3k.txt
│   ├── dev_1k.txt
│   └── test_1k.txt
│
├── config.yaml
├── config.py
├── dataset.py
├── model.py
├── train.py
├── README.md

二、各文件作用
config.yaml
用于保存项目中的配置参数 将路径和超参数放在配置文件中

config.py
负责读取 config.yaml 并将配置参数封装成配置类 供项目其他模块调用

dataset.py
负责构建新闻文本数据集
主要工作包括：
1.读取新闻数据
2.解析文本和标签
3.建立统一的标签映射
4.使用 BertTokenizer 对文本进行编码
5.返回 input_ids、attention_mask 和 label
6.通过 collate_fn 对一个 Batch 内的文本进行动态 Padding

model.py
定义 BERT 文本分类模型
模型主要由：
BERT
 ↓
Dropout
 ↓
Linear
组成

train.py
负责整个模型训练和评估流程 包括：
1.创建 Dataset
2.创建 DataLoader
3.创建模型
4.创建优化器
5.创建损失函数
6.模型训练
7.验证集评估
8.测试集评估
9.SwanLab 实验记录

三、数据处理流程
数据处理的大致流程：
原始新闻数据
      ↓
读取文件
      ↓
解析 label 和 text
      ↓
建立统一 label_map
      ↓
文本 Tokenizer
      ↓
得到 input_ids
      ↓
得到 attention_mask
      ↓
Dataset
      ↓
DataLoader
      ↓
collate_fn
      ↓
Batch 动态 Padding
      ↓
输入 BERT

四、模型训练
程序会依次完成：
读取配置文件
    ↓
建立统一的 label_map
    ↓
创建 Dataset
    ↓
创建 DataLoader
    ↓
创建 BERT 文本分类模型
    ↓
创建损失函数和 AdamW 优化器
    ↓
进入 Epoch 循环
    ↓
训练集进行一个 Epoch 的训练
    ↓
在 dev 验证集上计算 Accuracy
    ↓
重复上述训练 + 验证过程，直到完成所有 Epoch
    ↓
使用训练结束后的模型
    ↓
在 test 测试集上计算 Accuracy
    ↓
输出最终 Test Accuracy