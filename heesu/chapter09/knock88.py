import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knock87 import MODEL_PATH

SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]
LABELS = {0: "negative", 1: "positive"}


@torch.no_grad()
def main():
    # Load the model fine-tuned in knock87.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    inputs = tokenizer(SENTENCES, padding=True, return_tensors="pt")
    probs = torch.softmax(model(**inputs).logits, dim=-1)
    preds = probs.argmax(dim=-1)

    for sentence, pred, prob in zip(SENTENCES, preds, probs):
        label = LABELS[pred.item()]
        print(f"{label:8s} (p={prob[pred].item():.4f}) | {sentence}")


if __name__ == "__main__":
    main()
