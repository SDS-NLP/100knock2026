import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")
    model.eval()
    with torch.inference_mode():
        return model(**inputs).logits.argmax(dim=1).tolist()


def main():
    sentences = [
        "The movie was full of incomprehensibilities.",
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]
    model_dir = os.path.join(os.path.dirname(__file__), "models", "knock87")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    predictions = predict(sentences, tokenizer, model)

    for sentence, prediction in zip(sentences, predictions):
        label = "positive" if prediction == 1 else "negative"
        print(f"{label}: {sentence}")


if __name__ == "__main__":
    main()
