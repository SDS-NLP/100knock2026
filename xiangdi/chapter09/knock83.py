# 83. CLSトークンによる文ベクトル
# 以下の文の全ての組み合わせに対して、
# 最終層の[CLS]トークンの埋め込みベクトルを用いてコサイン類似度を求めよ。
import itertools
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

inputs = tokenizer(sentences, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

cls_vectors = outputs.last_hidden_state[:, 0, :]

for i, j in itertools.combinations(range(len(sentences)), 2):
    similarity = torch.nn.functional.cosine_similarity(
        cls_vectors[i],
        cls_vectors[j],
        dim=0
    )

    print(f"{sentences[i]}\t{sentences[j]}\t{similarity.item():.6f}")