import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

text = "The movie was full of [MASK]."
inputs = tokenizer(text, return_tensors="pt")   # PyTorchテンソルで返す
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits                    # (1, 系列長, 30522)

# [MASK] の位置を探す
mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

prob = torch.softmax(logits[0, mask_pos], dim=-1)
top = prob.topk(10)
for prob, idx in zip(top.values[0], top.indices[0]):
    token = tokenizer.decode(idx)
    print(f"{token:15s} {prob.item():.4f}")