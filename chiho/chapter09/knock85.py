"""85. データセットの準備

GLUE ベンチマークで配布されている SST から訓練セット（train.tsv）と
開発セット（dev.tsv）のテキストと極性ラベルを読み込み、
全てのテキストをトークン列に変換する。
"""

from dataclasses import dataclass
from pathlib import Path
import csv

from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
DATA_DIR = Path(__file__).resolve().parent.parent / "chapter07" / "data" / "SST-2"


@dataclass
class Example:
    text: str
    label: int
    token_ids: list[int]


def load_tsv(tsv_path: Path) -> list[tuple[str, int]]:
    # SST-2 の tsv ファイルから (文, ラベル) の組を読み込む
    examples = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return examples
        for row in reader:
            if not row:
                continue
            text, label = row
            examples.append((text.strip(), int(label)))
    return examples


def tokenize_examples(examples: list[tuple[str, int]]) -> list[Example]:
    # BERTのトークナイザで各文をトークンID列に変換する
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return [
        Example(text=text, label=label, token_ids=tokenizer.encode(text))
        for text, label in examples
    ]


def main() -> None:
    for split_name in ("train", "dev"):
        raw_examples = load_tsv(DATA_DIR / f"{split_name}.tsv")
        examples = tokenize_examples(raw_examples)
        print(f"{split_name}.tsv: {len(examples)} examples")
        for example in examples[:3]:
            print(f"  label={example.label} tokens={example.token_ids}")
            print(f"    text: {example.text}")


if __name__ == "__main__":
    main()
