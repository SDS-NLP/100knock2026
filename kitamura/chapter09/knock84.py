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

mean_vectors = {}
model.eval() # 推論に切り替え

with torch.no_grad(): # 勾配計算なし
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs) # 辞書の中身を展開して渡す
        mean_vector = outputs.last_hidden_state.mean(dim=1)
        mean_vectors[sentence] = mean_vector

# combinationsで重複のないすべてのペアを作る
for sent1, sent2 in itertools.combinations(sentences, 2):
    vector1 = mean_vectors[sent1]
    vector2 = mean_vectors[sent2]
    
    # コサイン類似度の計算
    similarity = F.cosine_similarity(vector1, vector2, dim=1).item()
    
    print(f"文章1: \"{sent1}\"")
    print(f"文章2: \"{sent2}\"")
    print(f"コサイン類似度: {similarity:.4f}")

"""文章1: "The movie was full of fun."
文章2: "The movie was full of excitement."
コサイン類似度: 0.9568
文章1: "The movie was full of fun."
文章2: "The movie was full of crap."
コサイン類似度: 0.8490
文章1: "The movie was full of fun."
文章2: "The movie was full of rubbish."
コサイン類似度: 0.8169
文章1: "The movie was full of excitement."
文章2: "The movie was full of crap."
コサイン類似度: 0.8352
文章1: "The movie was full of excitement."
文章2: "The movie was full of rubbish."
コサイン類似度: 0.7938
文章1: "The movie was full of crap."
文章2: "The movie was full of rubbish."
コサイン類似度: 0.9226"""