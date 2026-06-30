"""75. パディング

複数の事例をまとめて 1 つのテンソルにする collate 関数を実装する。
- トークン列の長さを最長のものに揃え、ID 0 (<PAD>) でパディングする。
- トークン列の長い事例から順に並び替える。

戻り値は {'input_ids': (batch, max_len), 'label': (batch, 1)} の辞書。
"""

import torch

from knock70 import PAD_ID


def collate(batch):
    """事例のリストをパディング済みのバッチテンソルにまとめる。"""
    # トークン列が長い順に並び替える
    batch = sorted(batch, key=lambda ex: ex["input_ids"].size(0), reverse=True)
    max_len = batch[0]["input_ids"].size(0)

    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        input_ids[i, : ids.size(0)] = ids

    labels = torch.stack([ex["label"] for ex in batch])  # (batch, 1)
    return {"input_ids": input_ids, "label": labels}


if __name__ == "__main__":
    # 問題文の冒頭4事例を再現して collate の動作を確認する
    examples = [
        {"text": "hide new secretions from the parental units",
         "label": torch.tensor([0.0]),
         "input_ids": torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594])},
        {"text": "contains no wit , only labored gags",
         "label": torch.tensor([0.0]),
         "input_ids": torch.tensor([3475, 87, 15888, 90, 27695, 42637])},
        {"text": "that loves its characters and communicates something rather beautiful about human nature",
         "label": torch.tensor([1.0]),
         "input_ids": torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964])},
        {"text": "remains utterly satisfied to remain the same throughout",
         "label": torch.tensor([0.0]),
         "input_ids": torch.tensor([987, 14528, 4941, 873, 12, 208, 898])},
    ]

    result = collate(examples)
    print("input_ids:")
    print(result["input_ids"])
    print("label:")
    print(result["label"])
