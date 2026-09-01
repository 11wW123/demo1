import torch

class Config:

    model_name="bert-base-chinese"

    train_path="./data/train_3k.txt"

    dev_path="./data/dev_1k.txt"

    test_path="./data/test_1k.txt"

    num_labels=15

    batch_size=16

    epochs=10

    lr=2e-5

    max_length=128 # 输入文本经过Tokenizer分词后 最多保留128个token（词元）

    device = "cuda" if torch.cuda.is_available() else "cpu"

config=Config()