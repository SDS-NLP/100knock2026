#"The movie was full of incomprehensibilities."という文をトークンに分解し、トークン列を表示せよ。

from transformers import BertTokenizer

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)

text = "The movie was full of incomprehensibilities."

tokens = tokenizer.tokenize(text)

if __name__ == "__main__":
    
    print(tokens)