import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import config

class ToutiaoDataset(Dataset):

    def __init__(self,file_path):

        self.data=[]   # 用来存新闻数据

        self.tokenizer=BertTokenizer.from_pretrained(config.model_name)

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            # 遍历每一行数据
            for line in f:
                # strip() 去掉前后多余的空白、换行
                # split() 以_!_为分隔符拆数据
                parts=line.strip().split("_!_")

                if len(parts)>=4:

                    label=int(parts[1])

                    text=parts[3]

                    self.data.append(
                        (
                            text,
                            label
                        )
                    )

        # 标签重新编号

        # sorted() 排序
        labels=sorted(
            # 转列表
            list(
                # set() 去重
                set(
                    # 把self.data里面每一条数据依次拿出来 叫做x 然后取它的第二项label
                    x[1]
                    for x in self.data
                )
            )
        )

        '''
        原始标签     新标签
        100     →     0
        101     →     1
        102     →     2
        103     →     3
               ...
        '''
        self.label_map={
            l:i
            for i,l in enumerate(labels)
        }

        '''
        new_data = []

        for text, label in self.data:
            new_label = self.label_map[label]

            new_data.append(
                (text, new_label)
            )
            
        self.data = new_data
        '''
        self.data=[
            (
                text,
                self.label_map[label]
            )
            for text,label in self.data
        ]



    def __len__(self):

        return len(self.data)


    # 取第index条数据 一次只处理一条文本 DataLoader会自动组batch
    def __getitem__(self,index):

        text,label=self.data[index]

        # 把中文新闻转换成BERT可以接收的数字数据
        encode=self.tokenizer(
            text, # 把刚才取出来的新闻文本交给Tokenizer
            max_length=config.max_length, # 规定文本最大长度
            padding="max_length", # 不足补
            truncation=True, # 超过截断
            return_tensors="pt" # 把结果转换成PyTorch Tensor
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