from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path("models")
MODEL_1_PATH = MODEL_PATH / "model_1.pth"
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model_1 = AutoModelForSequenceClassification.from_pretrained(model_name)
loaded_model_1.load_state_dict(
    torch.load(f=MODEL_1_PATH, map_location=torch.device("cpu"))
)
sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]
torch.manual_seed(42)
loaded_model_1.eval()
for sentence in sentences:
    with torch.inference_mode():
        tokens = tokenizer(
            sentence, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]
        dev_logits = loaded_model_1(tokens).logits
        dev_pred = torch.argmax(dev_logits, dim=1)
        print(f"{sentence}->{dev_pred}")
# Result
# The movie was full of incomprehensibilities.->tensor([0])
# The movie was full of fun.->tensor([1])
# The movie was full of excitement.->tensor([1])
# The movie was full of crap.->tensor([0])
# The movie was full of rubbish.->tensor([0])
