from gensim.models import KeyedVectors
import numpy as np


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vec1 = model['United_States']
vec2 = model['U.S.']

similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(similarity)
