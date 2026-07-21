from transformers import AutoTokenizer

# BERTのトークナイザーを読み込む
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The movie was full of incomprehensibilities."

tokens = tokenizer.tokenize(text)

print(tokens)

"""tokenizer_config.json: 100%|█████████████████████████| 48.0/48.0 [00:00<00:00, 117kB/s]
vocab.txt: 100%|████████████████████████████████████| 232k/232k [00:00<00:00, 1.33MB/s]
tokenizer.json: 100%|███████████████████████████████| 466k/466k [00:00<00:00, 1.39MB/s]
['the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.']"""