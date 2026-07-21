import torch

from knock70 import build_embedding_matrix
from knock71 import DEV_PATH, TRAIN_PATH, build_dataset, load_sst2
from knock72 import LogisticRegression
from knock73 import train_model

# 74. モデルの評価
# 学習済みモデルの、開発セット(dev)における正解率を求める。


def accuracy(model, data):
    model.eval()  # 評価モード(作法)
    correct = 0
    with torch.no_grad():  # 評価中は勾配を記録しない(学習しないので不要・速い)
        for ex in data:
            prob = model(ex["input_ids"])          # 確率 (1,)
            pred = 1 if prob.item() >= 0.5 else 0   # 0.5 で切って予測ラベル 0/1
            # 生の確率 prob ではなく、切った後の pred を正解ラベルと比べる
            if pred == ex["label"].item():
                correct += 1
    return correct / len(data)


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()
    train = build_dataset(load_sst2(TRAIN_PATH), word2id)
    dev = build_dataset(load_sst2(DEV_PATH), word2id)

    model = LogisticRegression(E)
    train_model(model, train)  # 73の学習を再利用
    print("dev accuracy:", accuracy(model, dev))
