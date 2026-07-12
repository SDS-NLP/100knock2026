from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "The movie was full of incomprehensibilities."

tokens = tokenizer.tokenize(text)
print(tokens)

print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))


# ['the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.']
# ['[CLS]', 'the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.', '[SEP]']
