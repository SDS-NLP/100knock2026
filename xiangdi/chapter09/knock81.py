import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "The movie was full of [MASK]."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_logits = logits[0, mask_token_index, :]

predicted_token_id = torch.argmax(mask_logits, dim=-1)
predicted_token = tokenizer.decode(predicted_token_id.item())

print(predicted_token)