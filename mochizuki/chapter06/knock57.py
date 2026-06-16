from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

countries = []

with open("questions-words.txt", "r", encoding = "utf-8") as file:
    for line in file:
        line = line.strip()
        if line.startswith(":"):
            if line == ": capital-common-countries" or line == ": capital-world":
                continue
            else: 
                break
        line = line.split()
        
        country1 = line[1]
        country2 = line[3]
        if country1 not in countries:
            countries.append(country1)
        if country2 not in countries:
            countries.append(country2)      
            
vectors = []

for country in countries:
    vector = model[country]
    vectors.append(vector)
country_vectors = np.array(vectors)

kmeans = KMeans(n_clusters = 5, random_state = 0, n_init = 10)
labels = kmeans.fit_predict(country_vectors)

if __name__ == "__main__":
    for country, label in zip(countries, labels):
        print(country, label)