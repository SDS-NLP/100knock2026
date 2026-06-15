from gensim.models import KeyedVectors
import numpy as np
model = KeyedVectors.load_word2vec_format(f'C:/study/NLP100knock/100knock2026/GoogleNews-vectors-negative300.bin.gz', binary=True)

spain = model["Spain"]
madrid = model["Madrid"]
Athens = model["Athens"]

target = spain - madrid + Athens
sim_list = []

for key in model.index_to_key:
    value = model[key]
    cos_sim = np.dot(value, target) / (np.linalg.norm(value) * np.linalg.norm(target))
    sim_list.append((key, cos_sim))

sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
for i in range(10):
    print(f"{sim_list[i][0]}: cos_sim {sim_list[i][1]}")