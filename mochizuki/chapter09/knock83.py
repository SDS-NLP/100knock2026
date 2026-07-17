"""
83. CLSトークンによる文ベクトル
以下の文の全ての組み合わせに対して、最終層の[CLS]トークンの埋め込みベクトルを用いてコサイン類似度を求めよ。

“The movie was full of fun.”

“The movie was full of excitement.”

“The movie was full of crap.”

“The movie was full of rubbish.”


"""


import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

embeddings = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
    embeddings.append(cls_embedding)

embeddings = np.array(embeddings)

similarity_matrix = cosine_similarity(embeddings)

print("文の組み合わせに対するコサイン類似度:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(
            f"'{sentences[i]}' と '{sentences[j]}' の類似度: {similarity_matrix[i][j]:.4f}"
        )

"""
ModernBertModel LOAD REPORT from: answerdotai/ModernBERT-base
Key               | Status     |  | 
------------------+------------+--+-
decoder.bias      | UNEXPECTED |  | 
head.dense.weight | UNEXPECTED |  | 
head.norm.weight  | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
文の組み合わせに対するコサイン類似度:
'The movie was full of fun.' と 'The movie was full of excitement.' の類似度: 0.9954
'The movie was full of fun.' と 'The movie was full of crap.' の類似度: 0.9912
'The movie was full of fun.' と 'The movie was full of rubbish.' の類似度: 0.9966
'The movie was full of excitement.' と 'The movie was full of crap.' の類似度: 0.9808
'The movie was full of excitement.' と 'The movie was full of rubbish.' の類似度: 0.9977
'The movie was full of crap.' と 'The movie was full of rubbish.' の類似度: 0.9857
"""