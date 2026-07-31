import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from knock85 import tokenize


def collate(batch):
    X = [item[0] for item in batch]
    y = [item[1] for item in batch]
    max_len = max(len(x) for x in X)
    for i in range(len(X)):
        X[i] = X[i] + [0] * (max_len - len(X[i]))
    return torch.tensor(X), torch.tensor(y, dtype=torch.float)


if __name__ == "__main__":
    TRAIN_PATH = "SST-2/train.tsv"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    df_train = tokenize(TRAIN_PATH, tokenizer)
    X_train = df_train.get_column("input_ids").to_list()
    y_train = df_train.get_column("label").to_list()
    train_dataset = list(zip(X_train, y_train))
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    for batch_X, batch_y in train_loader:
        print(f"Batch_X's shape:\n{batch_X.shape}")
        print(f"Batch_y's shape:\n{batch_y.shape}")
        break
# Result:
# Batch_X's shape:
# torch.Size([64, 72])
# Batch_y's shape:
# torch.Size([64])
