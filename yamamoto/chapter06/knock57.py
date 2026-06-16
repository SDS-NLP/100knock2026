#国名に関する単語ベクトルを抽出し、k-meansクラスタリングをクラスタ数k=5として実行せよ。

from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

countries = [] #国名を格納するリスト

with open("questions-words.txt", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        line = line.strip()
        
        if line.startswith(":"):
            
            if line == ": capital-common-countries" or line == ": capital-world": #国名があるセクションのときループ続行
                
                continue
            
            else:
                
                break
        
        line = line.split()
        
        country1 = line[1] #2つ目が国名
        country2 = line[3] #4つ目が国名
        
        if country1 not in countries: #まだ国名リストにないとき追加
            
            countries.append(country1)
        
        if country2 not in countries:
            
            countries.append(country2)      
            
vectors = [] #国名のベクトル表現を格納するリスト

for country in countries:
    
    vector = model[country]
    
    vectors.append(vector)

country_vectors = np.array(vectors)

kmeans = KMeans(n_clusters = 5, random_state = 0, n_init = 10) #n_cluster:クラスタ数, random_state:ランダム性, n_init:試行回数
labels = kmeans.fit_predict(country_vectors) #予測結果だけ欲しいときはfit_predict

if __name__ == "__main__":

    for country, label in zip(countries, labels): #zip()で2つのリストから同時に取り出す
    
        print(country, label)