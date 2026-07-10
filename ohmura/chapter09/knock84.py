import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import itertools

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

inputs = tokenizer(sentences, padding=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

mean_embeddings = torch.mean(outputs.last_hidden_state, dim=1)

pairs = list(itertools.combinations(range(len(sentences)), 2))

print("=== コサイン類似度の計算結果（平均ベクトル） ===\n")
for i, j in pairs:
    sim = F.cosine_similarity(mean_embeddings[i].unsqueeze(0), mean_embeddings[j].unsqueeze(0))
    
    print(f"文{i+1} と 文{j+1} の類似度: {sim.item():.4f}")
    print(f"  A: {sentences[i]}")
    print(f"  B: {sentences[j]}\n")