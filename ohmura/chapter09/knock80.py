from transformers import BertTokenizer

# 事前学習済みモデル（bert-base-uncased）のトークナイザをロード
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 対象のテキストをトークンに分解
print(tokenizer.tokenize("The movie was full of incomprehensibilities."))