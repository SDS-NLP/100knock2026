"""86. ミニバッチの作成

85で読み込んだ訓練データの一部（冒頭の4事例）に対して、
パディングを行いトークン列の長さを揃えてミニバッチを構成する。
"""

import torch
from transformers import AutoTokenizer

from knock85 import DATA_DIR, load_tsv


MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 4


def build_mini_batch(texts: list[str]) -> dict[str, torch.Tensor]:
    # 文の長さを最長のものに揃えてパディングし、ミニバッチのテンソルを作る
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(texts, padding=True, return_tensors="pt")


def main() -> None:
    raw_examples = load_tsv(DATA_DIR / "train.tsv")
    texts = [text for text, _ in raw_examples[:BATCH_SIZE]]
    labels = [label for _, label in raw_examples[:BATCH_SIZE]]

    batch = build_mini_batch(texts)

    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"labels: {labels}")
    for text, input_ids, attention_mask in zip(
        texts, batch["input_ids"], batch["attention_mask"]
    ):
        print(f"  text: {text}")
        print(f"    input_ids:      {input_ids.tolist()}")
        print(f"    attention_mask: {attention_mask.tolist()}")


if __name__ == "__main__":
    main()
