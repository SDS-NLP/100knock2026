import torch
import torch.nn as nn

from knock70 import build_embedding_matrix
from knock71 import TRAIN_PATH, build_dataset, load_sst2

# 75. パディング
# 複数事例 → 1個のテンソルにまとめる collate。
#   - 長さ降順ソート / 最長に合わせて id=0 で右詰め / label は縦積み
# 注: 平均時に 0 パディングを長さに数えない処理(マスク)は使う側の責任。


def collate(batch):
    """[{text, label:(1,), input_ids:(可変長, long)}, ...] → まとめた1個の辞書。"""
    # 長さ降順ソート。input_ids と label のペアを保つため「事例=辞書」ごと動かす。
    batch = sorted(batch, key=lambda ex: ex["input_ids"].size(0), reverse=True)
    seqs = [ex["input_ids"] for ex in batch]
    labels = [ex["label"] for ex in batch]

    # 最長に揃えて 0 で右詰め → (B, L_max)。batch_first=True で (B, L_max)(既定は (L_max, B))。
    # 入力(long)の dtype は維持されるので Embedding にそのまま渡せる。
    input_ids = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)

    # (1,) を B個 縦積み → (B, 1)。cat だと (B,) に潰れて後段の BCELoss でブロードキャスト事故。
    label = torch.stack(labels)

    return {"input_ids": input_ids, "label": label}


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()
    train = build_dataset(load_sst2(TRAIN_PATH), word2id)

    # 冒頭4事例で problem.md の期待出力と突き合わせる。
    out = collate(train[:4])
    print("input_ids:\n", out["input_ids"])
    print("label:\n", out["label"])
    print("shape:", out["input_ids"].shape, out["label"].shape)

    # 長さ降順ソートが効いているか(各行の非パディング長が降順か)。
    lengths = [int((row != 0).sum()) for row in out["input_ids"]]
    assert lengths == sorted(lengths, reverse=True), "長さ降順ソートが効いていない"
