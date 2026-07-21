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

token_vectors = outputs.last_hidden_state
attention_mask = inputs["attention_mask"].unsqueeze(-1)
sentence_vectors = (token_vectors * attention_mask).sum(dim=1)
sentence_vectors = sentence_vectors / attention_mask.sum(dim=1)

for i, j in itertools.combinations(range(len(sentences)), 2):
    similarity = torch.nn.functional.cosine_similarity(
        sentence_vectors[i].unsqueeze(0),
        sentence_vectors[j].unsqueeze(0)
    )

    print(f"{sentences[i]}\t{sentences[j]}\t{similarity.item():.6f}")
