import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from knock85 import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def collate(batch, pad_id=0):
    input_ids = [ex["input_ids"] for ex in batch]
    labels = torch.stack([ex["label"] for ex in batch])
    padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()      # 実トークン=1, PAD=0
    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "label": labels,
    }

if __name__ == "__main__":
    train = load_dataset("SST-2/train.tsv", tokenizer)

    batch = collate(train[:4])                      # 冒頭の4事例
    print("input_ids:")
    print(batch["input_ids"])
    print("attention_mask:")
    print(batch["attention_mask"])
    print("label:")
    print(batch["label"])
    print("shape:", batch["input_ids"].shape)