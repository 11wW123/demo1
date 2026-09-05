import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import swanlab
from dataset import ToutiaoDataset
from model import BertClassifier
from config import config


def train_one_epoch(model,loader,optimizer,loss_fn):

    model.train()

    total_loss = 0

    for batch in tqdm(loader):

        ids = batch["input_ids"].to(
            config.device
        )

        mask = batch["attention_mask"].to(
            config.device
        )

        y = batch["labels"].to(
            config.device
        )

        optimizer.zero_grad()

        output = model(
            ids,
            mask
        )

        loss = loss_fn(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss


def evaluate(model,loader):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in loader:

            ids = batch["input_ids"].to(
                config.device
            )

            mask = batch["attention_mask"].to(
                config.device
            )

            y = batch["labels"].to(
                config.device
            )

            out = model(
                ids,
                mask
            )

            pred = torch.argmax(
                out,
                dim=1
            )

            # 统计预测正确的数量
            correct += (pred == y).sum().item()

            # 统计总样本数量
            total += y.size(0)

    # 计算准确率
    return correct / total


def main():

    swanlab.init(

        project="demo1-bert-text-classification",

        config={

            "lr": config.lr,

            "batch_size": config.batch_size,

            "epochs": config.epochs
        }
    )


    # 生成统一的标签映射表
    label_map = ToutiaoDataset.build_label_map(
        config.train_path
    )

    print(
        "Label Map:",
        label_map
    )

    train_dataset = ToutiaoDataset(
        config.train_path,
        label_map
    )

    dev_dataset = ToutiaoDataset(
        config.dev_path,
        label_map
    )

    test_dataset = ToutiaoDataset(
        config.test_path,
        label_map
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.batch_size,

        shuffle=True,

        collate_fn=train_dataset.collate_fn
    )


    dev_loader = DataLoader(

        dev_dataset,

        batch_size=config.batch_size,

        collate_fn=dev_dataset.collate_fn
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=config.batch_size,

        collate_fn=test_dataset.collate_fn
    )

    model = BertClassifier().to(config.device)


    optimizer = AdamW(

        model.parameters(),

        lr=config.lr
    )


    loss_fn = torch.nn.CrossEntropyLoss()


    for epoch in range(
        config.epochs
    ):

        total_loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            loss_fn
        )


        dev_acc = evaluate(

            model,

            dev_loader
        )


        print(
            f"""
Epoch:{epoch + 1}
loss:{total_loss:.4f}
dev_acc:{dev_acc:.4f}
"""
        )


        swanlab.log({

            "loss": total_loss,

            "dev_acc": dev_acc

        })


    test_acc = evaluate(

        model,

        test_loader
    )


    print(

        "Test Accuracy:",

        test_acc
    )


    swanlab.log({

        "test_acc": test_acc

    })


    swanlab.finish()


if __name__ == "__main__":

    main()