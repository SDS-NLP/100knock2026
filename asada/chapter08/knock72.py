import torch
from torch import nn

from knock70 import WordEmbeddingToolkit
from knock71 import tokenize


class ModelV0(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class ModelV1(nn.Module):
    def __init__(self, input_dim, pre_weights, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pre_weights, dtype=torch.float32),
            freeze=freeze,
            padding_idx=0,
        )
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        x_embedded = self.embedding(x).mean(dim=1)
        return self.linear(x_embedded)


class ModelV2(nn.Module):
    def __init__(self, input_dim, hidden_units, pre_weights, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pre_weights, dtype=torch.float32),
            freeze=freeze,
            padding_idx=0,
        )
        self.block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units, out_features=1),
        )

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    toolkit = WordEmbeddingToolkit()
    TRAIN_PATH = "SST-2/train.tsv"
    DEV_PATH = "SST-2/dev.tsv"
    df_train = tokenize(TRAIN_PATH, toolkit)
    df_dev = tokenize(DEV_PATH, toolkit)

    X_train = torch.tensor(df_train.get_column("mean_vector").to_list())
    X_dev = torch.tensor(df_dev.get_column("mean_vector").to_list())
    y_train = torch.tensor(df_train.get_column("label").to_list())
    y_dev = torch.tensor(df_dev.get_column("label").to_list())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = X_train.shape[1]
    model_0 = ModelV0(input_dim).to(device)
    model_1 = ModelV1(input_dim, toolkit.matrix).to(device)
    model_2 = ModelV2(input_dim, 5, toolkit.matrix).to(device)
