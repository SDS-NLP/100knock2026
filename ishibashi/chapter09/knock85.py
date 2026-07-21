import csv
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "bert-base-uncased"
DATASET_DIR = Path(__file__).with_name("SST-2")


def load_split(tsv_path: Path, tokenizer: PreTrainedTokenizerBase) -> tuple[list[str], list[int], list[list[str]]]:
    """SST-2のTSVを読み込み、全テキストをBERTのWordPieceへ分割する。"""
    texts: list[str] = []
    labels: list[int] = []
    token_lists: list[list[str]] = []

    with tsv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            text = row["sentence"]
            texts.append(text)
            labels.append(int(row["label"]))
            token_lists.append(tokenizer.tokenize(text))
    return texts, labels, token_lists


def show_split(name: str, texts: list[str], labels: list[int], tokens: list[list[str]]) -> None:
    print(f"{name}: {len(texts)} examples")
    for text, label, token_list in zip(texts[:3], labels[:3], tokens[:3]):
        print(f"  label={label}\ttext={text}\n  tokens={token_list}")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels, train_tokens = load_split(DATASET_DIR / "train.tsv", tokenizer)
    dev_texts, dev_labels, dev_tokens = load_split(DATASET_DIR / "dev.tsv", tokenizer)
    show_split("train", train_texts, train_labels, train_tokens)
    show_split("dev", dev_texts, dev_labels, dev_tokens)


if __name__ == "__main__":
    main()