import torch
import torch.nn as nn
from knock70 import build_embedding_matrix
from knock71 import TRAIN_PATH, build_dataset, load_sst2
from knock72 import LogisticRegression

# 73. モデルの学習
# 訓練セット上で fc の重みを学習する(埋め込みは freeze 済みで固定)。
# 学習ループ = (w-5)^2 デモと同じ骨: 予測→損失→勾配→更新 を繰り返す。

LR = 0.01
EPOCHS = 10


def train_model(model, train_data, lr=LR, epochs=EPOCHS):
    """訓練セット上で fc の重みを学習する(埋め込みは freeze 済みで動かない)。"""
    criterion = nn.BCELoss()  # 72はsigmoid済みの確率を返す → BCELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for ex in train_data:
            prob = model(ex["input_ids"])         # 予測(確率) (1,)
            loss = criterion(prob, ex["label"])   # 正解ラベルとのズレ

            # 更新の3手: 前回の勾配を消す → 勾配を計算 → 勾配の逆向きに重みを更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        # 進捗モニタリング: エポック平均の損失(下がれば学習できている)
        print(f"epoch {epoch}: loss = {total_loss / len(train_data):.4f}")
    return model


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()
    train = build_dataset(load_sst2(TRAIN_PATH), word2id)

    model = LogisticRegression(E)
    train_model(model, train)
