import sys
from pathlib import Path

from knock80 import load_tokenizer

# 85. データセットの準備
# SST-2 の train/dev からテキストと極性ラベルを読み、全テキストをID列に変換する。
# 読み込みは chap7 の load_sst2 を再利用(パスは chapter07 相対なのでここで組み直す)。
# ID列は素の長さのまま持つ。パディングはバッチを作る86でやる。

CHAP07 = Path(__file__).resolve().parent.parent / "chapter07"
sys.path.insert(0, str(CHAP07))
from sst2_common import load_sst2  # noqa: E402

TRAIN_PATH = CHAP07 / "SST-2" / "train.tsv"
DEV_PATH = CHAP07 / "SST-2" / "dev.tsv"


def load_dataset(path, tokenizer):
    """{text, label(int), input_ids} の辞書リスト。86以降で使う。"""
    data = load_sst2(path)
    ids = tokenizer([text for text, _ in data])["input_ids"]
    return [
        {"text": text, "label": int(label), "input_ids": i}
        for (text, label), i in zip(data, ids)
    ]


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    train = load_dataset(TRAIN_PATH, tokenizer)
    dev = load_dataset(DEV_PATH, tokenizer)

    # ラベルは int の 0/1 だけ、ID列は [CLS]...[SEP] の形になっているか。
    assert {d["label"] for d in train} == {0, 1}, "ラベルが0/1以外"
    assert all(
        d["input_ids"][0] == tokenizer.cls_token_id
        and d["input_ids"][-1] == tokenizer.sep_token_id
        for d in train
    ), "特殊トークンが付いていない"

    print("train:", len(train))  # 67349
    print("dev:", len(dev))      # 872
    for d in train[:3]:
        print(d["label"], d["text"])
        print(tokenizer.convert_ids_to_tokens(d["input_ids"]))
