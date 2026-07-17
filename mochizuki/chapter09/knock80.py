"""
80. トークン化
“The movie was full of incomprehensibilities.”という文をトークンに分解し、トークン列を表示せよ。
"""

from transformers import AutoTokenizer

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "The movie was full of incomprehensibilities."
inputs = tokenizer(text, return_tensors="pt")
print(inputs)

"""
{'input_ids': tensor([[50281,   510,  6440,   369,  2120,   273, 15321,  8391,  9594,    15,
         50282]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
"""