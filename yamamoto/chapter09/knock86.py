#85で読み込んだ訓練データの一部（例えば冒頭の4事例）に対して、パディングなどの処理を行い、トークン列の長さを揃えてミニバッチを構成せよ。

from transformers import BertTokenizer
import torch
import knock85

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train = knock85.train

examples = train[:4]

sentences = [example["sentence"] for example in examples]
labels = [example["label"] for example in examples]

batch = tokenizer(
    sentences,
    padding = True,
    truncation = True,
    return_tensors = "pt"
)

labels = torch.tensor(labels)

if __name__ == "__main__":
    
    print("input_ids:")
    print(batch["input_ids"])

    print("attention_mask:")
    print(batch["attention_mask"])

    print("labels:")
    print(labels)

    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", labels.shape)