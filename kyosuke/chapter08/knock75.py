import torch
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    # 1. トークン列の長い順に事例を並び替える
    batch = sorted(batch, key=lambda ex: len(ex["input_ids"]), reverse=True)

    # 2. 並び替え後の input_ids と label を取り出す
    input_ids = [ex["input_ids"] for ex in batch]
    labels = [ex["label"] for ex in batch]

    # 3. 最長に合わせて 0 でパディングし、まとめて1つのテンソルに
    padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {"input_ids": padded, "label": labels}