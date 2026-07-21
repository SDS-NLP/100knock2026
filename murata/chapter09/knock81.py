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

pred_id = logits[0, mask_idx].argmax().item()
print(tokenizer.decode([pred_id]))
