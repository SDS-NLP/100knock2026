"""69. 正則化パラメータの変更

ロジスティック回帰モデルを学習するとき、正則化係数 C を変化させながら
検証データ上の正解率を求め、結果をグラフにまとめる。
"""

from __future__ import annotations

import math
import numpy as np

from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from knock61 import DATA_DIR, load_dataset


OUTPUT_PATH = DATA_DIR / "knock69_accuracy.svg"


def vectorize_data(train_data, dev_data):
    vectorizer = DictVectorizer(sparse=True)
    train_x = sparse.csr_matrix(vectorizer.fit_transform([d["feature"] for d in train_data]))
    dev_x = sparse.csr_matrix(vectorizer.transform([d["feature"] for d in dev_data]))

    train_x.indices = train_x.indices.astype("int32", copy=False)
    train_x.indptr = train_x.indptr.astype("int32", copy=False)
    dev_x.indices = dev_x.indices.astype("int32", copy=False)
    dev_x.indptr = dev_x.indptr.astype("int32", copy=False)
    return train_x, dev_x


def train_and_score(train_x, train_y, dev_x, dev_y, c_value: float) -> float:
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42, C=c_value)
    model.fit(train_x, train_y)
    return accuracy_score(dev_y, model.predict(dev_x))


def build_svg(c_values: list[float], accuracies: list[float]) -> str:
    width = 800
    height = 500
    margin_left = 70
    margin_right = 20
    margin_top = 40
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_c = math.log10(min(c_values))
    max_c = math.log10(max(c_values))
    min_acc = min(accuracies) - 0.02
    max_acc = max(accuracies) + 0.02

    def x_pos(c_value: float) -> float:
        ratio = (math.log10(c_value) - min_c) / (max_c - min_c)
        return margin_left + ratio * plot_width

    def y_pos(acc: float) -> float:
        ratio = (acc - min_acc) / (max_acc - min_acc)
        return margin_top + (1.0 - ratio) * plot_height

    points = " ".join(f"{x_pos(c):.2f},{y_pos(a):.2f}" for c, a in zip(c_values, accuracies))

    x_labels = "\n".join(
        f'<text x="{x_pos(c):.2f}" y="{height - 35}" text-anchor="middle" font-size="12">{c:g}</text>'
        for c in c_values
    )
    y_ticks = []
    for i in range(6):
        acc = min_acc + i * (max_acc - min_acc) / 5
        y = y_pos(acc)
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#ddd" />'
        )
        y_ticks.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{acc:.3f}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="sans-serif">Logistic Regression Regularization</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="black" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="black" />
  {''.join(y_ticks)}
  <polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{points}" />
  {''.join(f'<circle cx="{x_pos(c):.2f}" cy="{y_pos(a):.2f}" r="4" fill="#1f77b4" />' for c, a in zip(c_values, accuracies))}
  <text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-size="14" font-family="sans-serif">C</text>
  <text x="18" y="{height / 2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 18 {height / 2})">dev accuracy</text>
  {x_labels}
</svg>
"""


def main() -> None:
    train_data = load_dataset(DATA_DIR / "train.tsv")
    dev_data = load_dataset(DATA_DIR / "dev.tsv")

    train_x, dev_x = vectorize_data(train_data, dev_data)
    train_y = [int(d["label"]) for d in train_data]
    dev_y = [int(d["label"]) for d in dev_data]

    c_values = np.logspace(-2, 2, 9)
    accuracies = []
    for c_value in c_values:
        accuracy = train_and_score(train_x, train_y, dev_x, dev_y, c_value)
        accuracies.append(accuracy)
        print(f"C={c_value:<5} dev accuracy={accuracy:.6f}")

    OUTPUT_PATH.write_text(build_svg(c_values, accuracies), encoding="utf-8")
    print(f"saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
