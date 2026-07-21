from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import torch.nn.functional as F
from itertools import combinations

MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL)
text = "The movie was full of incomprehensibilities."
encoded_text = tokenizer.tokenize(text)
print(encoded_text)


