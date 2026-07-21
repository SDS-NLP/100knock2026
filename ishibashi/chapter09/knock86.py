import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TRAIN_PATH = Path(__file__).with_name("SST-2") / "train.tsv"
BATCH_SIZE = 4


def load_first_examples(tsv_path: Path, count: int) -> tuple[list[str], list[int]]:
    """TSVの先頭count件について、テキストと極性ラベルを読み込む。"""
    texts: list[str] = []
    labels: list[int] = []
    with tsv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            texts.append(row["sentence"])
            labels.append(int(row["label"]))
            if len(texts) == count:
                break
    return texts, labels


def main() -> None:
    texts, labels = load_first_examples(TRAIN_PATH, BATCH_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # padding=True により、この4文の最大長まで [PAD] を付与する。
    encoded_batch = tokenizer(texts, padding=True, return_tensors="pt")
    batch = {
        "input_ids": encoded_batch["input_ids"],
        "attention_mask": encoded_batch["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    print("labels:", batch["labels"].tolist())
    for index, (text, token_ids, mask) in enumerate(
        zip(texts, batch["input_ids"], batch["attention_mask"]), start=1
    ):
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        print(f"\nexample {index}: {text}")
        print("tokens:         ", tokens)
        print("attention_mask: ", mask.tolist())


if __name__ == "__main__":
    main()