import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from knock80 import MODEL_NAME, load_tokenizer
from knock85 import DEV_PATH, TRAIN_PATH, load_dataset
from knock86 import make_batch

# 87. ファインチューニング
# 分類ヘッド(768→2)付きBERTをSST-2で学習し、devセットで正解率を測る。
# ヘッドはランダム初期化から始まる(ロード時のMISSINGは今回は意図どおり)。
# lossは自分で計算する(86で inputs と labels を分けた設計の回収)。
# M1(8GB)のMPSで回すため既定は8000件に間引く。N_TRAIN=None でフル67k件。
#
# 結果(8000件・seed0): dev_acc 0.874 → 0.907 → 0.905。chap7のBoW+LogReg(訓練67k)は 0.812。
# 表示のlossは訓練lossなので下がり続ける(epoch3で0.059)。dev_accとの連動が切れたら過学習。
# 保存は最終エポックの重み。dev最良(epoch2)を保存する early stopping は入れていない。

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_TRAIN = 8000
BATCH_SIZE = 32
N_EPOCHS = 3
LR = 5e-5  # ファインチューニングは小さく。大きいと事前学習の重みを壊す
SAVE_DIR = "sst2-model"


def load_classifier(model_name=MODEL_NAME):
    """分類ヘッド付きBERT。88でも使う。"""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def make_loader(data, pad_id, batch_size=BATCH_SIZE, shuffle=False):
    """辞書リストをそのまま渡し、バッチ化は86の make_batch に任せる。"""
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(make_batch, pad_id=pad_id))


def accuracy(model, loader):
    """argmax予測の正解率。88/89でも使う。"""
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            out = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
            logits = out.logits if hasattr(out, "logits") else out  # 89の自作モデルは生テンソル
            correct += (logits.argmax(dim=-1).cpu() == labels).sum().item()
            n += len(labels)
    return correct / n


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    train = load_dataset(TRAIN_PATH, tokenizer)
    dev = load_dataset(DEV_PATH, tokenizer)
    if N_TRAIN:
        train = random.Random(0).sample(train, N_TRAIN)

    model = load_classifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train_loader = make_loader(train, tokenizer.pad_token_id, shuffle=True)
    dev_loader = make_loader(dev, tokenizer.pad_token_id)

    for epoch in range(N_EPOCHS):
        model.train()
        total = 0.0
        for inputs, labels in train_loader:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            loss = F.cross_entropy(model(**inputs).logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)

        acc = accuracy(model, dev_loader)
        print(f"epoch {epoch + 1}: loss={total / len(train):.4f} dev_acc={acc:.4f}")

    model.save_pretrained(SAVE_DIR)  # 88で読み直す。重みはコミットしない
    tokenizer.save_pretrained(SAVE_DIR)
