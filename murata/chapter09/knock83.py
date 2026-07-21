from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import torch.nn.functional as F
from itertools import combinations

MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL)
model = BertModel.from_pretrained(MODEL).eval()

sents = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

cls_vecs = []
with torch.no_grad():
    for s in sents:
        enc = tokenizer(s, return_tensors="pt")
        out = model(**enc)
        cls_vecs.append(out.last_hidden_state[0, 0])  # [CLS] は先頭

for (i, a), (j, b) in combinations(enumerate(cls_vecs), 2):
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"{sents[i]} <-> {sents[j]}: {cos:.4f}")