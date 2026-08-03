import csv
import re
from pathlib import Path


def simple_tokenize(text: str) -> list[str]:
    """
    簡易トークナイザ
    ・小文字化
    ・単語（英数字）と記号を分割してトークン列にする
    ※ 外部ライブラリ（nltkなど）を使わずに実装
    """
    text = text.lower()
    # 単語 or 記号1文字 をトークンとして抽出
    tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]", text)
    return tokens


def load_tsv(file_path: str):
    """
    tsvファイルを読み込み、(トークン列, ラベル) のリストを返す

    戻り値:
        data: [{"text": str, "tokens": list[str], "label": str}, ...]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # 1行目がヘッダー（"sentence" "label" など）かどうかを簡易判定
    start_idx = 0
    if rows and len(rows[0]) >= 2:
        # ラベル列が数値でなければヘッダー行とみなす
        if not rows[0][1].strip().isdigit():
            start_idx = 1

    for row in rows[start_idx:]:
        if len(row) < 2:
            # 列数が足りない行はスキップ
            continue
        text, label = row[0], row[1]
        tokens = simple_tokenize(text)
        data.append({
            "text": text,
            "tokens": tokens,
            "label": label,
        })

    return data


def main():
    # 訓練セットと開発セットを読み込む
    train_data = load_tsv("train.tsv")
    dev_data = load_tsv("dev.tsv")

    print(f"訓練セット件数: {len(train_data)}")
    print(f"開発セット件数: {len(dev_data)}")

    # サンプルとして先頭3件を表示
    print("\n--- 訓練セットのサンプル ---")
    for sample in train_data[:3]:
        print(f"テキスト: {sample['text']}")
        print(f"トークン: {sample['tokens']}")
        print(f"ラベル  : {sample['label']}")
        print("-" * 40)


if __name__ == "__main__":
    main()