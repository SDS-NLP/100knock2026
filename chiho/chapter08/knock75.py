"""75. パディング

複数の事例をまとめて一つのテンソル・オブジェクトで表現する関数 collate を実装する。
トークン列の長さが異なるときは最長のものに揃え、0 番のトークン ID でパディングする。
さらに、トークン列の長さが長いものから順に事例を並び替える。
"""

from __future__ import annotations

import torch

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets


def pad_to_length(tensor: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
    """1 次元テンソルを target_len の長さになるまで pad_value で埋める。

    例: tensor([3475, 87]) を長さ 4 に揃える -> tensor([3475, 87, 0, 0])
    """
    # まず「全部 pad_value で埋まった長さ target_len のテンソル」を作る
    # (dtype を元のテンソルに合わせることで、整数 ID が float にならないようにする)
    padded = torch.full((target_len,), pad_value, dtype=tensor.dtype)

    # 先頭部分に元のテンソルの中身を上書きコピーする
    # 残った後ろの部分は pad_value (= 0) のまま残るので、これがパディングになる
    padded[: tensor.size(0)] = tensor

    return padded


def collate(batch: list[dict[str, object]], pad_value: int = 0) -> dict[str, torch.Tensor]:
    """複数の事例をまとめて一つのテンソルで表現する。

    手順:
    1. トークン列 (input_ids) が長い順に事例を並び替える
    2. 最長のトークン列に合わせて、短い列の後ろを pad_value で埋める
    3. torch.stack で 1 つのテンソル (バッチ) にまとめる
    """
    # 1. input_ids の長さが長い順 (降順) に並び替える
    sorted_batch = sorted(batch, key=lambda example: example["input_ids"].size(0), reverse=True)

    # 2. 並び替え後の先頭が最長なので、その長さをパディングの目標長にする
    max_len = sorted_batch[0]["input_ids"].size(0)

    # 3. 各事例の input_ids を最長に揃えてから、torch.stack で
    #    (バッチサイズ, 最長トークン数) の 2 次元テンソルに積み上げる
    input_ids = torch.stack(
        [pad_to_length(example["input_ids"], max_len, pad_value) for example in sorted_batch]
    )

    # label は元々どれも同じ形 (要素数 1) なので、そのまま積み上げるだけでよい
    # 結果は (バッチサイズ, 1) の 2 次元テンソルになる
    labels = torch.stack([example["label"] for example in sorted_batch])

    return {"input_ids": input_ids, "label": labels}


def main() -> None:
    _, token_to_id, _ = load_embedding_matrix()
    train_dataset, _ = load_train_dev_datasets(token_to_id)

    # 問題文の例と同じ、訓練データセットの冒頭 4 事例で動作を確認する
    head = train_dataset[:4]

    print("--- collate 前 ---")
    for example in head:
        print(example)

    print("--- collate 後 ---")
    batch = collate(head)
    print(batch)
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"label shape: {batch['label'].shape}")


if __name__ == "__main__":
    main()
