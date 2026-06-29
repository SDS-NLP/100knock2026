from transformers import pipeline

model_name = "bert-base-uncased"
unmasker = pipeline("fill-mask", model=model_name)

text = "The movie was full of [MASK]."

results = unmasker(text)

best_result = results[0]
print(best_result["token_str"])

# fun