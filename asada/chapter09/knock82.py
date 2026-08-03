from transformers import pipeline

text = "The movie was full of [MASK]."
pipeline = pipeline("fill-mask", model="bert-base-cased", top_k=10)
prediction = pipeline(text)
print(prediction)
# [{'score': 0.04128894582390785, 'token': 22810, 'token_str': 'surprises',
#  'sequence': 'The movie was full of surprises.'}, {'score': 0.03167409449
# 81575, 'token': 1390, 'token_str': 'music', 'sequence': 'The movie was fu
# ll of music.'}, {'score': 0.029457353055477142, 'token': 4170, 'token_str
# ': 'shit', 'sequence': 'The movie was full of shit.'}, {'score': 0.024980
# 181828141212, 'token': 1234, 'token_str': 'people', 'sequence': 'The movi
# e was full of people.'}, {'score': 0.0232686847448349, 'token': 4106, 'to
# ken_str': 'fun', 'sequence': 'The movie was full of fun.'}, {'score': 0.0
# 21825453266501427, 'token': 1122, 'token_str': 'it', 'sequence': 'The mov
# ie was full of it.'}, {'score': 0.020956652238965034, 'token': 12375, 'to
# ken_str': 'laughs', 'sequence': 'The movie was full of laughs.'}, {'score
# ': 0.0204338189214468, 'token': 7053, 'token_str': 'laughter', 'sequence'
# : 'The movie was full of laughter.'}, {'score': 0.01903490163385868, 'tok
# en': 1172, 'token_str': 'them', 'sequence': 'The movie was full of them.'
# }, {'score': 0.01770380698144436, 'token': 11074, 'token_str': 'crap', 's
# equence': 'The movie was full of crap.'}]
