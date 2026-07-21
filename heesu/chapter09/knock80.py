from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
SENTENCE = "The movie was full of incomprehensibilities."


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Sub-word tokens produced by the WordPiece tokenizer (no special tokens).
    tokens = tokenizer.tokenize(SENTENCE)
    print("Tokens:", tokens)

    # The same sequence with the special [CLS]/[SEP] tokens BERT actually sees.
    encoded = tokenizer(SENTENCE)
    with_special = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    print("With special tokens:", with_special)
    print("Input IDs:", encoded["input_ids"])


if __name__ == "__main__":
    main()
