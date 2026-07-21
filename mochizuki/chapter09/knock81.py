"""
81. マスクの予測
“The movie was full of [MASK].”の”[MASK]”を埋めるのに最も適切なトークンを求めよ。
"""

from transformers import pipeline
from pprint import pprint

pipe = pipeline("fill-mask", model="answerdotai/ModernBERT-base", device="cpu", top_k=1)

input_text = "The movie was full of [MASK]."
results = pipe(input_text)
pprint(results)

"""
[{'score': 0.05700276419520378,
  'sequence': 'The movie was full of surprises.',
  'token': 37700,
  'token_str': ' surprises'}]
"""