from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
text = "The movie was full of [MASK]."

result = unmasker(text)

top = result[0]
print(top["token_str"])
print(top["score"])

# fun
# 0.10711907595396042
