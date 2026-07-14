"""
82. マスクのtop-k予測
“The movie was full of [MASK].”の”[MASK]”に埋めるのに適切なトークン上位10個と、その確率（尤度）を求めよ。
"""


from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask", model="answerdotai/ModernBERT-base", device="cpu", top_k=10
)

input_text = "The movie was full of [MASK]."
results = pipe(input_text)
pprint(results)

"""
[{'score': 0.05700276419520378,
  'sequence': 'The movie was full of surprises.',
  'token': 37700,
  'token_str': ' surprises'},
 {'score': 0.040078505873680115,
  'sequence': 'The movie was full of controversy.',
  'token': 16305,
  'token_str': ' controversy'},
 {'score': 0.036364637315273285,
  'sequence': 'The movie was full of errors.',
  'token': 6332,
  'token_str': ' errors'},
 {'score': 0.028496719896793365,
  'sequence': 'The movie was full of fun.',
  'token': 794,
  'token_str': ' fun'},
 {'score': 0.025707177817821503,
  'sequence': 'The movie was full of humor.',
  'token': 20393,
  'token_str': ' humor'},
 {'score': 0.02353915013372898,
  'sequence': 'The movie was full of problems.',
  'token': 3237,
  'token_str': ' problems'},
 {'score': 0.022886056452989578,
  'sequence': 'The movie was full of mistakes.',
  'token': 16503,
  'token_str': ' mistakes'},
 {'score': 0.017484622076153755,
  'sequence': 'The movie was full of love.',
  'token': 2389,
  'token_str': ' love'},
 {'score': 0.016137540340423584,
  'sequence': 'The movie was full of holes.',
  'token': 11385,
  'token_str': ' holes'},
 {'score': 0.014168394729495049,
  'sequence': 'The movie was full of flaws.',
  'token': 32138,
  'token_str': ' flaws'}]
"""