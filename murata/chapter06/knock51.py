from gensim.models import KeyedVectors
import numpy as np
model = KeyedVectors.load_word2vec_format(f'C:/study/NLP100knock/100knock2026/GoogleNews-vectors-negative300.bin.gz', binary=True)

united = model["United_States"]
US = model["U.S."]

print(np.dot(united, US)/ (np.linalg.norm(united) * np.linalg.norm(US)))
