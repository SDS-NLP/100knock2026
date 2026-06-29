import itertools
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    "The movie was full of fun.",       # ポジティブ
    "The movie was full of excitement.",# ポジティブ
    "The movie was full of crap.",      # ネガティブ
    "The movie was full of rubbish."    # ネガティブ
]

# すべての文章の [CLS] ベクトルを予め計算して辞書に格納
cls_vectors = {}
model.eval()

with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        # 最終層の隠れ状態から先頭の [CLS] トークン（インデックス0）を抽出
        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vectors[sentence] = cls_vector

# combinations(データ, 2) で重複のないすべてのペアを作ります
for sent1, sent2 in itertools.combinations(sentences, 2):
    vector1 = cls_vectors[sent1]
    vector2 = cls_vectors[sent2]
    
    # コサイン類似度の計算
    similarity = F.cosine_similarity(vector1, vector2, dim=1).item()
    
    print(f"文章1: \"{sent1}\"")
    print(f"文章2: \"{sent2}\"")
    print(f"コサイン類似度: {similarity:.4f}")
    
"""
文章1: "The movie was full of fun."
文章2: "The movie was full of excitement."
コサイン類似度: 0.9881
文章1: "The movie was full of fun."
文章2: "The movie was full of crap."
コサイン類似度: 0.9558
文章1: "The movie was full of fun."
文章2: "The movie was full of rubbish."
コサイン類似度: 0.9475
文章1: "The movie was full of excitement."
文章2: "The movie was full of crap."
コサイン類似度: 0.9541
文章1: "The movie was full of excitement."
文章2: "The movie was full of rubbish."
コサイン類似度: 0.9487
文章1: "The movie was full of crap."
文章2: "The movie was full of rubbish."
コサイン類似度: 0.9807"""