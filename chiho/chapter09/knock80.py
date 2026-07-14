"""80. トークン化

文をBERTのWordPieceトークナイザでトークンに分解して表示する。
"""

from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TEXT = "The movie was full of incomprehensibilities."


def tokenize(text: str) -> list[str]:
    # BERTの語彙を使って文をWordPieceトークンに分割する
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer.tokenize(text)


def main() -> None:
    tokens = tokenize(TEXT)
    print(tokens)


if __name__ == "__main__":
    main()
