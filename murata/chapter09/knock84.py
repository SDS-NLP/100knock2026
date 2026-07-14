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

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

mean_vecs = []
with torch.no_grad():
    for s in sents:
        enc = tokenizer(s, return_tensors="pt")
        out = model(**enc)
        vec = mean_pool(out.last_hidden_state, enc.attention_mask)[0]
        mean_vecs.append(vec)

for (i, a), (j, b) in combinations(enumerate(mean_vecs), 2):
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"{sents[i]} <-> {sents[j]}: {cos:.4f}")