from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "The movie was full of incomprehensibilities."
    tokens = tokenizer.tokenize(text)
    print(tokens)

if __name__ == "__main__":
    main()