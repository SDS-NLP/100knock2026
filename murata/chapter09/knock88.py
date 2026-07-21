import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL)
model = BertForSequenceClassification.from_pretrained('bert_sst_finetuned').eval()

sents = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

with torch.no_grad():
    enc = tokenizer(sents, return_tensors="pt", padding=True)
    probs = F.softmax(model(**enc).logits, dim=-1)

for s, p in zip(sents, probs):
    label = 'positive' if p[1] > p[0] else 'negative'
    print(f'{s} -> {label} (neg={p[0]:.4f}, pos={p[1]:.4f})')
