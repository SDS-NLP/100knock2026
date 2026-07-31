import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

cls_embeddings = []
for sentence in sentences:
    tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    model.eval()
    with torch.inference_mode():
        output = model(**tokens)
    cls_embedding = output.last_hidden_state[0][0].numpy()
    cls_embeddings.append(cls_embedding)

similarity_matrix = cosine_similarity(np.array(cls_embeddings))

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(
            f"{sentences[i]}と{sentences[j]}のコサイン類似度: {similarity_matrix[i][j]:.3f}"
        )
# The movie was full of fun.とThe movie was full of excitement.のコサイン類
# 似度: 0.989
# The movie was full of fun.とThe movie was full of crap.のコサイン類似度:
# 0.979
# The movie was full of fun.とThe movie was full of rubbish.のコサイン類似
# 度: 0.975
# The movie was full of excitement.とThe movie was full of crap.のコサイン
# 類似度: 0.967
# The movie was full of excitement.とThe movie was full of rubbish.のコサイ
# ン類似度: 0.965
# The movie was full of crap.とThe movie was full of rubbish.のコサイン類似
# 度: 0.994
