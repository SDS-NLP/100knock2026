#以下の文の全ての組み合わせに対して、最終層の[CLS]トークンの埋め込みベクトルを用いてコサイン類似度を求めよ。
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
    truncation = True, #1文の最大長(512トークン)に収まるように切る
    return_tensors = "pt"
)

with torch.no_grad():
    
    outputs = model(**inputs) #inputの各トークンを埋め込みに変換(encode)

last_hidden_state = outputs.last_hidden_state #最終層における各トークンの埋め込み(bert-base-uncasedでは第12層)

cls_embedding = last_hidden_state[:, 0, :] #CLS:文の先頭に付されるトークンで、その埋め込みは文全体の特徴を表すことが期待される(分類など文全体について何か判断するときに使われる), last_hidden_state[バッチサイズ(文の数),トークン数,各トークンの埋め込みの次元数]

if __name__ == "__main__":
    
    for i, j in combinations(range(len(sentences)), 2): #リストから2つづつ取り出す
    
        sim = F.cosine_similarity(
            cls_embedding[i].unsqueeze(0),
            cls_embedding[j].unsqueeze(0)
        ).item()
    
        print(f"{i + 1} - {j + 1}: {sim:.4f}")
        print(sentences[i])
        print(sentences[j])
        print()