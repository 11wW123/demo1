import torch.nn as nn

from transformers import BertModel

from config import config

class BertClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert=BertModel.from_pretrained(
            config.model_name
        )

        # 防止模型过拟合
        self.dropout=nn.Dropout(
            0.3
        )

        # 分类器 768 -> 15
        # bert-base 的隐藏层大小就是768
        self.fc=nn.Linear(
            768,
            config.num_labels
        )



    def forward(self,input_ids,attention_mask):

        """
        output 是 BertModel 的输出对象
        其中 output.pooler_output 是我们取出来用于分类的部分
        它是一个768维向量 也就是768个数字（一条新闻数据对应768个数字）
        这768个数字不是768个类别 而是模型对这条文本语义的数学表示
        后面的Linear层再把这768个数字转换成15个类别分数
        """
        output=self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        cls=output.pooler_output


        x=self.dropout(cls)


        logits=self.fc(x)

        # 把模型最终得到的15个类别分数返回
        return logits