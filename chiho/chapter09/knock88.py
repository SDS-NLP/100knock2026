"""88. 極性分析

87でファインチューニングしたモデルを用いて、与えられた文の極性を予測する。
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knock87 import MODEL_DIR


SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]
LABELS = {0: "negative", 1: "positive"}


def predict(sentences: list[str]) -> list[tuple[int, float]]:
    # ファインチューニング済みモデルで各文の極性ラベルと確率を予測する
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(sentences, padding=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)
    predicted_labels = probabilities.argmax(dim=-1)

    return [
        (int(label), float(probabilities[i, label]))
        for i, label in enumerate(predicted_labels)
    ]


def main() -> None:
    for sentence, (label, probability) in zip(SENTENCES, predict(SENTENCES)):
        print(f"{LABELS[label]} ({probability:.4f}): {sentence}")


if __name__ == "__main__":
    main()
