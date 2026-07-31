from transformers import pipeline

text = "The movie was full of [MASK]."
pipeline = pipeline("fill-mask", model="bert-base-cased", top_k=1)
prediction = pipeline(text)
print(prediction)
# [{'score': 0.04128894582390785, 'token': 22810, 'token_str': 'surpris
# es', 'sequence': 'The movie was full of surprises.'}]
