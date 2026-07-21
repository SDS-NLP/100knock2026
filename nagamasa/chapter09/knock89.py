import random

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from knock80 import MODEL_NAME, load_tokenizer
from knock85 import DEV_PATH, TRAIN_PATH, load_dataset
from knock87 import DEVICE, LR, N_EPOCHS, N_TRAIN, accuracy, make_loader

# 89. アーキテクチャの変更
# 87([CLS]系の既製ヘッド)に対し、最大値プーリングの分類器を自作して比較する。
# ヘッドの深さは87の既製ヘッド(dense+tanh → dropout → linear)に揃え、
# 差分が「[CLS] vs maxプーリング」だけになるようにする。
# データ・ハイパラは87と同一(seed0の8000件・lr5e-5・バッチ32・3エポック)。
# ただしバッチ32のままだとMPSが8GBを使い切る(masked_fillのコピー分87より重い)ため、
# マイクロバッチ16×勾配累積2で「メモリは16相当・勾配は32件分」にする。
# backwardは勾配を加算する仕様なので、stepとzero_gradを2回に1回にすれば累積になる。
#
# 結果: dev_acc 0.866 → 0.892 → 0.906。87([CLS]系・0.907)との差はdev872件中1文。
# 凍結ベクトルでは大差だった集約方式(83/84)も、ファインチューニング後はほぼ並ぶ。

MICRO_BATCH = 16
ACCUM = 2  # 実効バッチ = MICRO_BATCH × ACCUM = 32


class MaxPoolClassifier(nn.Module):
    """BERT本体 + 最大値プーリング + 87既製ヘッドと同構造の分類ヘッド。"""

    def __init__(self, model_name=MODEL_NAME, n_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.dense = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # PAD位置を -inf に置いてから各次元の最大値を取る (バッチ, 768)。
        # 84の平均は掛け算(×0)でPADを消せたが、maxでは0が最大値になり得るため -inf が要る。
        mask = attention_mask.unsqueeze(-1).bool()
        pooled = hidden.masked_fill(~mask, float("-inf")).max(dim=1).values

        pooled = torch.tanh(self.dense(pooled))
        return self.classifier(self.dropout(pooled))  # 生のlogits (バッチ, 2)


if __name__ == "__main__":
    torch.manual_seed(0)  # ヘッド初期値・シャッフル順・dropoutを固定

    tokenizer = load_tokenizer()
    train = load_dataset(TRAIN_PATH, tokenizer)
    dev = load_dataset(DEV_PATH, tokenizer)
    if N_TRAIN:
        train = random.Random(0).sample(train, N_TRAIN)  # 87と同じ8000件

    model = MaxPoolClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train_loader = make_loader(train, tokenizer.pad_token_id, batch_size=MICRO_BATCH, shuffle=True)
    dev_loader = make_loader(dev, tokenizer.pad_token_id, batch_size=MICRO_BATCH)

    for epoch in range(N_EPOCHS):
        model.train()
        total = 0.0
        optimizer.zero_grad()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            loss = F.cross_entropy(model(**inputs), labels)
            (loss / ACCUM).backward()  # 累積2回で32件分の平均勾配になるよう割る
            total += loss.item() * len(labels)

            if (step + 1) % ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

        acc = accuracy(model, dev_loader)
        print(f"epoch {epoch + 1}: loss={total / len(train):.4f} dev_acc={acc:.4f}")
