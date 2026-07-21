from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "bert_sst2_model"

sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

id_to_label = {
    0: "negative",
    1: "positive",
}


if not model_dir.exists():
    raise FileNotFoundError(
        f"{model_dir} does not exist. Run knock87.py first to fine-tune and save the model."
    )

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=-1)
predicted_label_ids = probabilities.argmax(dim=-1)

for sentence, label_id, probability in zip(
    sentences,
    predicted_label_ids,
    probabilities,
):
    label = id_to_label[label_id.item()]
    score = probability[label_id].item()

    print(f"{sentence}\t{label}\t{score:.6f}")

# The movie was full of incomprehensibilities.    negative        0.991362
# The movie was full of fun.      positive        0.995076
# The movie was full of excitement.       positive        0.990517
# The movie was full of crap.     negative        0.989778
# The movie was full of rubbish.  negative        0.991643