from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
text = "The movie was full of incomprehensibilities."
tokens = tokenizer(text)
print(tokens)
# トークン列の表示
# {'input_ids': [101, 1109, 2523,
#  1108, 1554, 1104, 1107, 8178,
# 1643, 1874, 10436, 5053, 15951,
#  119, 102], 'token_type_ids': [
# 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# , 0, 0, 0, 0], 'attention_mask'
# : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1
# , 1, 1, 1, 1, 1]}
