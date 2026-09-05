import yaml

class Config:

    def __init__(self, config_path="./config.yaml"):

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:

            config = yaml.safe_load(f)

        self.model_name = config["model_name"]

        self.train_path = config["train_path"]

        self.dev_path = config["dev_path"]

        self.test_path = config["test_path"]

        self.num_labels = config["num_labels"]

        self.batch_size = config["batch_size"]

        self.epochs = config["epochs"]

        self.lr = config["lr"]

        self.max_length = config["max_length"]

        self.device = "cuda"

config = Config()