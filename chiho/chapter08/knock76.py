"""76. ミニバッチ学習

問題75のパディングの処理を活用して、ミニバッチでモデルを学習する。
また、学習したモデルの開発セットにおける正解率を求める。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets
from knock72 import SentenceLogisticRegression
from knock73 import set_seed
from knock75 import collate


EPOCHS = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 8


def train_model_minibatch(
    model: SentenceLogisticRegression,
    train_loader: DataLoader,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> list[float]:
    """ミニバッチでモデルを学習する。

    問題73では 1 事例ずつパラメータを更新していたが、ここでは
    DataLoader が collate でまとめてくれたミニバッチ (複数事例) 単位で
    損失を計算し、まとめてパラメータを更新する。
    """
    # BCEWithLogitsLoss = シグモイド + 二値交差エントロピー損失
    # (モデルの出力がシグモイド適用前の logit なのでこちらを使う)
    criterion = nn.BCEWithLogitsLoss()

    # 単語埋め込みは固定 (freeze) されているので、線形層のパラメータだけ更新する
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=learning_rate)

    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()  # 訓練モードに切り替える
        total_loss = 0.0

        # train_loader から collate 済みのミニバッチが 1 つずつ出てくる
        # batch["input_ids"]: (バッチサイズ, 最長トークン数) にパディング済み
        # batch["label"]:     (バッチサイズ, 1)
        for batch in train_loader:
            input_ids = batch["input_ids"]
            # モデルの出力 logits は (バッチサイズ,) の形なので、
            # label も view(-1) で (バッチサイズ, 1) -> (バッチサイズ,) に揃える
            labels = batch["label"].view(-1)

            optimizer.zero_grad()  # 前のバッチの勾配をリセット
            logits = model(input_ids)  # バッチ全体をまとめて順伝播
            loss = criterion(logits, labels)  # バッチ平均の損失
            loss.backward()  # 逆伝播で勾配を計算
            optimizer.step()  # パラメータを更新

            # loss はバッチ内の平均なので、事例数を掛けて合計に戻して足し込む
            total_loss += float(loss.item()) * input_ids.size(0)

        # 全事例で平均した損失 (問題73と同じ基準で比較できるようにする)
        average_loss = total_loss / len(train_loader.dataset)
        loss_history.append(average_loss)
        print(f"epoch {epoch:02d} loss={average_loss:.6f}")

    return loss_history


def compute_accuracy_minibatch(
    model: SentenceLogisticRegression,
    data_loader: DataLoader,
) -> float:
    """ミニバッチ単位でまとめて予測し、正解率を求める。"""
    model.eval()  # 評価モードに切り替える
    correct = 0
    total = 0

    with torch.no_grad():  # 評価時は勾配計算を行わない
        for batch in data_loader:
            labels = batch["label"].view(-1)

            # 確率が 0.5 以上ならポジティブ (1.0)、未満ならネガティブ (0.0)
            probabilities = model.predict_proba(batch["input_ids"])
            predictions = (probabilities >= 0.5).float()

            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    return correct / total


def main() -> None:
    set_seed()
    embedding_matrix, token_to_id, _ = load_embedding_matrix()
    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)
    model = SentenceLogisticRegression(embedding_matrix)

    # DataLoader にデータセットと collate 関数を渡すと、
    # 自動でミニバッチを切り出して collate でテンソルにまとめてくれる
    # (訓練データは毎エポック順番をシャッフルする)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    train_model_minibatch(model, train_loader)

    train_accuracy = compute_accuracy_minibatch(model, train_loader)
    dev_accuracy = compute_accuracy_minibatch(model, dev_loader)
    print(f"train accuracy: {train_accuracy:.6f}")
    print(f"dev accuracy: {dev_accuracy:.6f}")


if __name__ == "__main__":
    main()
