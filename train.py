import torch

from torch.utils.data import DataLoader

from torch.optim import AdamW # 优化器 调整模型参数

from sklearn.metrics import accuracy_score # 计算准确率

from tqdm import tqdm # 显示训练进度条

import swanlab # 实验记录工具（loss下降曲线 accuracy变化） 方便观察训练效果



from dataset import ToutiaoDataset

from model import BertClassifier

from config import config


swanlab.init(

    project="demo1-bert-text-classification",

    config={

        "lr":config.lr,

        "batch_size":config.batch_size,

        "epochs":config.epochs

    }

)



train_dataset=ToutiaoDataset(
    config.train_path
)


dev_dataset=ToutiaoDataset(
    config.dev_path
)


test_dataset=ToutiaoDataset(
    config.test_path
)



train_loader=DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True # 随机打乱训练集数据
)


dev_loader=DataLoader(
    dev_dataset,
    batch_size=config.batch_size
)


test_loader=DataLoader(
    test_dataset,
    batch_size=config.batch_size
)


# .to()是PyTorch的函数 放到GPU计算
model=BertClassifier().to(
    config.device
)

# 创建一个AdamW优化器（AdamW是一种算法）
# 让它负责调整BertClassifier模型内部所有参数 每次调整幅度由学习率控制
optimizer=AdamW(
    model.parameters(), # 把模型里面所有需要学习的参数交给优化器管理
    lr=config.lr
)

# 交叉熵损失函数
loss_fn=torch.nn.CrossEntropyLoss()



def evaluate(loader):

    model.eval() # 切换模型状态为测试

    preds=[] # 保存模型预测结果

    labels=[] # 保存真实答案


    with torch.no_grad():

        for batch in loader:

            ids=batch["input_ids"].to(
                config.device
            )

            mask=batch["attention_mask"].to(
                config.device
            )

            y=batch["labels"].to(
                config.device
            )

            out=model(
                ids,
                mask
            )

            pred=torch.argmax(
                out,
                dim=1
            )


            preds.extend(
                pred.cpu().numpy()
            )

            labels.extend(
                y.cpu().numpy()
            )


    return accuracy_score(
        labels,
        preds
    )



# 真正训练开始 循环10次
for epoch in range(config.epochs):

    # 进入训练模式（PyTorch模型的状态切换）
    model.train()

    total_loss=0

    for batch in tqdm(train_loader):

        ids=batch["input_ids"].to(
            config.device
        )


        mask=batch["attention_mask"].to(
            config.device
        )


        y=batch["labels"].to(
            config.device
        )

        # PyTorch默认：梯度会累积
        # 每次训练前 清空梯度
        optimizer.zero_grad()



        output=model(
            ids,
            mask
        )


        loss=loss_fn(
            output,
            y
        )


        loss.backward()


        optimizer.step()



        total_loss+=loss.item()



    dev_acc=evaluate(dev_loader)


    print(
        f"""
Epoch:{epoch+1}
loss:{total_loss:.4f}
dev_acc:{dev_acc:.4f}
"""
    )


    swanlab.log({

        "loss":total_loss,

        "dev_acc":dev_acc

    })




test_acc=evaluate(
    test_loader
)


print(
    "Test Accuracy:",
    test_acc
)


swanlab.log(
    {
        "test_acc":test_acc
    }
)


swanlab.finish()