"""
SST-2 データセットを読み込み, テキストをトークンID列に変換するスクリプト.

語彙外の単語は無視し, 全トークンが語彙外となった事例は除外する.
前ステップで構築した word2id を使用する.
"""

import torch
import pandas as pd

from knock70 import build_embedding_matrix


# トークナイズ

def tokenize(text: str, word2id: dict[str, int]) -> list[int]:
    """
    テキストを語彙内トークンのID列に変換する.

    SST-2 は事前にスペース区切りでトークナイズ済みのため,
    空白分割のみで対応できる. 語彙外の単語はスキップする.
    """
    return [word2id[tok] for tok in text.split() if tok in word2id]


# データセット読み込み

def load_sst2(
    tsv_path: str,
    word2id: dict[str, int],
    split_name: str = "",
) -> list[dict]:
    """
    SST-2 形式の TSV を読み込み, 事例の辞書リストを返す.

    各事例は以下のキーを持つ辞書:
      - text      : 元のテキスト文字列
      - label     : 極性ラベル（ポジティブ: tensor([1.]), ネガティブ: tensor([0.])）
      - input_ids : 語彙内トークンのID列 (1-D LongTensor)

    全トークンが語彙外となった事例は除外される.

    Parameters
    ----------
    tsv_path : str
        読み込む TSV ファイルのパス（ヘッダ行あり, sentence/label カラム）.
    word2id : dict[str, int]
        単語からトークンIDへの対応辞書.
    split_name : str
        ログ表示用のデータセット名（省略可）.

    Returns
    -------
    list[dict]
        フィルタリング済みの事例リスト.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    dataset: list[dict] = []
    n_removed = 0

    for _, row in df.iterrows():
        text  = str(row["sentence"])
        label = int(row["label"])

        input_ids = tokenize(text, word2id)

        # 全トークンが語彙外 → 事例ごとスキップ
        if len(input_ids) == 0:
            n_removed += 1
            continue

        dataset.append(
            {
                "text":      text,
                "label":     torch.tensor([float(label)]),
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
            }
        )

    tag = f"[{split_name}] " if split_name else ""
    print(f"{tag}読み込み完了: {len(dataset):,} 件  （空列により除外: {n_removed} 件）")
    return dataset

if __name__ == "__main__":
    # 事前学習済み単語ベクトルから word2id を構築
    model_path = "GoogleNews-vectors-negative300.bin.gz"
    _, word2id, _ = build_embedding_matrix(model_path)

    # TSV パス（スクリプトからの相対パス）
    train_path = "SST-2/SST-2/train.tsv"
    dev_path   = "SST-2/SST-2/dev.tsv"

    train_dataset = load_sst2(train_path, word2id, split_name="train")
    dev_dataset   = load_sst2(dev_path,   word2id, split_name="dev")

    # 先頭3件を確認
    print("\n---- 訓練セット 先頭3件 ----")
    for example in train_dataset[:3]:
        print(example)