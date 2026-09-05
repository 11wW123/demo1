import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import config

class ToutiaoDataset(Dataset):

    def __init__(self,file_path,label_map):

        self.data=[]   # 用来存新闻数据

        self.tokenizer=BertTokenizer.from_pretrained(config.model_name)

        self.load_data(file_path)

        self.process_labels(label_map)


    def load_data(self, file_path):

        with open(
                file_path,
                "r",
                encoding="utf-8"
        ) as f:

            # 遍历每一行数据
            for line in f:
                # strip() 去掉前后多余的空白、换行
                # split() 以_!_为分隔符拆数据
                parts = line.strip().split("_!_")

                if len(parts) >= 4:
                    label = int(parts[1])

                    text = parts[3]

                    self.data.append(
                        (
                            text,
                            label
                        )
                    )


    def process_labels(self,label_map):

        self.label_map = label_map

        '''
        new_data = []

        for text, label in self.data:
            new_label = self.label_map[label]

            new_data.append(
                (text, new_label)
            )

        self.data = new_data
        '''
        self.data = [
            (
                text,
                self.label_map[label]
            )
            for text, label in self.data
        ]


    @staticmethod
    def build_label_map(file_path):

        # 用来保存训练集中的原始标签
        labels = []

        with open(
                file_path,
                "r",
                encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split("_!_")

                if len(parts) >= 4:

                    label = int(parts[1])

                    labels.append(label)

        # 去重并排序
        labels = sorted(
            set(labels)
        )

        '''
        原始标签     新标签
        100     →     0
        101     →     1
        102     →     2
        103     →     3
               ...
        '''
        label_map = {
            label: i
            for i, label in enumerate(labels)
        }

        return label_map


    def __len__(self):

        return len(self.data)


    # 取第index条数据 一次只处理一条文本 DataLoader会自动组batch
    def __getitem__(self,index):

        text,label=self.data[index]

        # 把中文新闻转换成BERT可以接收的数字数据
        encode=self.tokenizer(
            text,
            max_length=config.max_length,
            padding=False,
            truncation=True, # 超过截断
            return_tensors="pt"
        )

        '''
        最终返回一个字典 里面有三样东西
        '''
        return {
            # 新闻文字转换成的数字编号
            "input_ids":
                # .squeeze(0)：删除第0维（维度大小为1）
                encode["input_ids"].squeeze(0),

            # 告诉BERT：哪些数字是真正的文本 哪些是补出来的
            "attention_mask":
                encode["attention_mask"].squeeze(0),


            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }

    def collate_fn(self, batch):

        input_ids = [
            item["input_ids"]
            for item in batch
        ]

        attention_masks = [
            item["attention_mask"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )


        attention_masks = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0
        )

        labels = torch.stack(labels)

        return {

            "input_ids": input_ids,

            "attention_mask": attention_masks,

            "labels": labels

        }