"""
複数の事例をパディング・ソートしてバッチテンソルにまとめる関数の実装.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate(examples: list[dict]) -> dict:
    """
    複数の事例をパディングし, 単一のバッチテンソルにまとめる.

    トークン列を長さの降順にソートしたうえで, 最長に合わせて
    トークンID=0（<PAD>）でパディングする.

    Parameters
    ----------
    examples : list[dict]
        各要素は 'input_ids'（1-D LongTensor）と 'label'（Tensor([float])）を持つ辞書.

    Returns
    -------
    dict
        - 'input_ids' : Tensor (N, L_max)  パディング済みトークンID行列
        - 'label'     : Tensor (N, 1)      ラベル行列
    """
    # トークン列の長さで降順ソート
    examples = sorted(examples, key=lambda ex: len(ex["input_ids"]), reverse=True)

    input_ids_list = [ex["input_ids"] for ex in examples]
    label_list     = [ex["label"]     for ex in examples]

    # 最長に合わせて 0 でパディング（pad_sequence はデフォルトで列方向に揃える）
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels           = torch.stack(label_list)

    return {"input_ids": padded_input_ids, "label": labels}


if __name__ == "__main__":
    # 問題文の4事例で動作確認
    examples = [
        {
            "text":      "hide new secretions from the parental units",
            "label":     torch.tensor([0.]),
            "input_ids": torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594]),
        },
        {
            "text":      "contains no wit , only labored gags",
            "label":     torch.tensor([0.]),
            "input_ids": torch.tensor([3475, 87, 15888, 90, 27695, 42637]),
        },
        {
            "text":      "that loves its characters and communicates something rather beautiful about human nature",
            "label":     torch.tensor([1.]),
            "input_ids": torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964]),
        },
        {
            "text":      "remains utterly satisfied to remain the same throughout",
            "label":     torch.tensor([0.]),
            "input_ids": torch.tensor([987, 14528, 4941, 873, 12, 208, 898]),
        },
    ]

    batch = collate(examples)
    print(batch)