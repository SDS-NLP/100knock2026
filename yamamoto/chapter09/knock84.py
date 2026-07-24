#以下の文の全ての組み合わせに対して、最終層の埋め込みベクトルの平均を用いてコサイン類似度を求めよ。
#"The movie was full of fun."
#"The movie was full of excitement."
#"The movie was full of crap."
#"The movie was full of rubbish."

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from itertools import combinations

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name) #トークナイザー
model = BertModel.from_pretrained(model_name) #使用するBERTモデル

model.eval()

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

inputs = tokenizer(
    sentences,
    padding = True,
    truncation = True,
    return_tensors = "pt"
)

with torch.no_grad():
    
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state

attention_mask = inputs["attention_mask"] #実トークンなら1,[PAD]なら0

mask = attention_mask.unsqueeze(-1) #token_embeddingと掛け算できるように次元を1つ増やす

masked_embeddings = token_embeddings * mask #掛け算をすることでtoken_embeddingにおける[PAD]のベクトルを0にする

mean_embeddings = masked_embeddings.sum(dim = 1) / mask.sum(dim = 1) #各トークンのベクトルの合計/トークン数

for i, j in combinations(range(len(sentences)), 2):
    
    sim = F.cosine_similarity(
        mean_embeddings[i].unsqueeze(0),
        mean_embeddings[j].unsqueeze(0)
    ).item()
    
    print(f"{i + 1} - {j + 1}: {sim:.4f}")
    print(sentences[i])
    print(sentences[j])
    print()