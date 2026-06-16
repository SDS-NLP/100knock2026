"""61. 特徴量抽出

Bag of Words (BoW) に基づき、学習データ（train.tsv）および
検証データ（dev.tsv）のテキストを特徴ベクトルに変換する。
各事例は、テキスト・特徴ベクトル・ラベルを持つ辞書として表現する。
"""

from pathlib import Path
import csv
from pprint import pprint


DATA_DIR = Path(__file__).resolve().parent / "data" / "SST-2"


def make_feature(text: str) -> dict[str, int]:
    # BoW 特徴ベクトルに変換
    feature: dict[str, int] = {}
    for token in text.split():
        feature[token] = feature.get(token, 0) + 1
    return feature


def load_dataset(tsv_path: Path) -> list[dict[str, object]]:
    # データセットのロード
    dataset: list[dict[str, object]] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            text = row[0].strip()
            label = row[1].strip()
            dataset.append(
                {
                    "text": text,
                    "label": label,
                    "feature": make_feature(text),
                }
            )
    return dataset


def main() -> None:
    train_data = load_dataset(DATA_DIR / "train.tsv")
    dev_data = load_dataset(DATA_DIR / "dev.tsv")

    print(f"train size: {len(train_data)}")
    print(f"dev size: {len(dev_data)}")
    print("first train example:")
    pprint(train_data[0], sort_dicts=False)


if __name__ == "__main__":
    main()
