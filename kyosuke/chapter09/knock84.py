import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from itertools import combinations

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

inputs = tokenizer(sentences, padding=True, return_tensors="pt")

with torch.no_grad():
    last_hidden = model(**inputs).last_hidden_state      # (4, seq_len, 768)

mask = inputs["attention_mask"].unsqueeze(-1)            # (4, seq_len) → (4, seq_len, 1)
summed = (last_hidden * mask).sum(dim=1)                 # PAD位置を0にしてから和 (4, 768)
lengths = mask.sum(dim=1)                                # 実トークン数 (4, 1)
mean_vecs = summed / lengths                             # (4, 768)

for i, j in combinations(range(len(sentences)), 2):
    sim = F.cosine_similarity(mean_vecs[i], mean_vecs[j], dim=0).item()
    print(f"{sim:.4f}  |  {sentences[i]}  vs  {sentences[j]}")