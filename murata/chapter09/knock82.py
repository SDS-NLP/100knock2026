from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import torch.nn.functional as F
from itertools import combinations

MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL)
mlm = BertForMaskedLM.from_pretrained(MODEL).eval()

text = "The movie was full of [MASK]."
enc = tokenizer(text, return_tensors="pt")
mask_idx = (enc.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

with torch.no_grad():
    logits = mlm(**enc).logits

probs = F.softmax(logits[0, mask_idx], dim=-1)
topk = torch.topk(probs, k=10)

for prob, idx in zip(topk.values, topk.indices):
    print(f"{tokenizer.decode([idx.item()]):15s} {prob.item():.4f}")