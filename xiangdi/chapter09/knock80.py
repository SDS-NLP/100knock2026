from transformers import AutoTokenizer

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "The movie was full of incomprehensibilities."

tokens = tokenizer.tokenize(text)
print(tokens)