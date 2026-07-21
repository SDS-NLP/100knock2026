import csv

import torch
from knock70 import build_embedding_matrix

# 71. データセットの読み込み
# SST-2 の train/dev を読み、各テキストをトークンID列に変換する。
#   - OOV(word2id に無い語)は列に入れない。
#   - 全語OOVで空列になった事例は削除する。
#   - 各事例を {text, label(tensor), input_ids(tensor)} で表す。

TRAIN_PATH = "../chapter07/SST-2/train.tsv"
DEV_PATH = "../chapter07/SST-2/dev.tsv"


def load_sst2(path):
    # 7章 sst2_common.load_sst2 の複製(自前方式を選択)。
    # 返り値: [(sentence, label), ...]  label は文字列 "0"/"1"。先頭行はヘッダ。
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # ヘッダ "sentence\tlabel" を飛ばす
        return [(sentence, label) for sentence, label in reader]


def text_to_ids(text, word2id):
    """空白分かち書き → ID列。OOV(word2id に無い語)は入れない。返り値: int の list。"""
    # `if w in word2id` で OOV を弾く(.get(w, 0) は OOV を PAD=0 に化けさせるので不可)。
    return [word2id[w] for w in text.split() if w in word2id]


def build_dataset(data, word2id):
    """(sentence, label) のリスト → [{text, label, input_ids}, ...]。空列の事例は捨てる。"""
    dataset = []
    for sentence, label in data:
        ids = text_to_ids(sentence, word2id)
        if not ids:  # 全語OOVで空列になった事例は除外
            continue
        dataset.append(
            {
                "text": sentence,
                # label は float: 損失で出力(float)と比較する量。元データは文字列なので float() を噛ます。
                "label": torch.tensor([float(label)]),
                # input_ids は long: 埋め込み行列 E の行番号(索引)として使うため。
                "input_ids": torch.tensor(ids, dtype=torch.long),
            }
        )
    return dataset


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()

    raw_train = load_sst2(TRAIN_PATH)
    raw_dev = load_sst2(DEV_PATH)
    train = build_dataset(raw_train, word2id)
    dev = build_dataset(raw_dev, word2id)

    print("train size:", len(train), "/ raw:", len(raw_train))
    print("dev size:  ", len(dev), "/ raw:", len(raw_dev))
    # 70で先送りした宿題: 空列で何件落ちたか(=被覆率の体感)
    print("dropped (empty) train:", len(raw_train) - len(train))
    print("example[0]:", train[0])
