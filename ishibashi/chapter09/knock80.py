from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TEXT = "The movie was full of incomprehensibilities."


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer.tokenize(TEXT)
    print(tokens)


if __name__ == "__main__":
    main()