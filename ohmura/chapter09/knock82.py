import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The movie was full of [MASK]."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

mask_logits = outputs.logits[0, mask_token_index, :]

probabilities = F.softmax(mask_logits, dim=-1)[0]

top10_probs, top10_indices = torch.topk(probabilities, 10)

for i, (prob, idx) in enumerate(zip(top10_probs, top10_indices), 1):
    token = tokenizer.decode([idx])
    print(f"{i:2d}: {token:<12} 確率: {prob.item():.4f}")