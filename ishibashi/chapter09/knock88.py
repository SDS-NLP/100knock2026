"""第9章 knock88: ファインチューニング済みBERTで極性を予測する。"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).with_name("fine_tuned_bert_sst2")
SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]
LABEL_NAMES = {0: "negative", 1: "positive"}


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"{MODEL_DIR} was not found. Run knock87.py first.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    inputs = tokenizer(SENTENCES, padding=True, truncation=True, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1).cpu()

    for text, probability in zip(SENTENCES, probabilities):
        label_id = probability.argmax().item()
        print(
            f"{LABEL_NAMES[label_id]}\t"
            f"negative={probability[0]:.4f}\tpositive={probability[1]:.4f}\t{text}"
        )


if __name__ == "__main__":
    main()
