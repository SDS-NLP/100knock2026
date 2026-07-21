from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def main():
    sentence = "The movie was full of incomprehensibilities."
    tokenizer = load_tokenizer()
    tokens = tokenizer.tokenize(sentence)

    print(tokens)


if __name__ == "__main__":
    main()
