import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_dataset(path, tokenizer):
    data = []
    with open(path, encoding="utf-8") as f:
        next(f)                                    # ヘッダを読み飛ばす(71と同じ)
        for line in f:
            text, label = line.rstrip("\n").split("\t")
            enc = tokenizer(text)                  # トークンID列に変換
            data.append({
                "text": text,
                "label": torch.tensor([float(label)]),
                "input_ids": torch.tensor(enc["input_ids"]),
            })
    return data

if __name__ == "__main__":
    train = load_dataset("SST-2/train.tsv", tokenizer)
    dev = load_dataset("SST-2/dev.tsv", tokenizer)
    print(f"train: {len(train)} 件, dev: {len(dev)} 件")

    # 確認: 先頭の1件を人間可読な形で表示
    ex = train[0]
    print("text:      ", ex["text"])
    print("input_ids: ", ex["input_ids"])
    print("tokens:    ", tokenizer.convert_ids_to_tokens(ex["input_ids"]))
    print("label:     ", ex["label"])