#The WordSimilarity-353 Test Collectionの評価データをダウンロードし、単語ベクトルにより計算される類似度のランキングと、人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ。

from gensim.models import KeyedVectors
import numpy as np
from scipy.stats import spearmanr

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

word_list = []
line_count = 0

with open("combined.tab", "r", encoding = "utf-8") as file:
    
    for line in file:
        
        line_count += 1
        
        if line_count == 1:
            
            continue
            
        line = line.strip()
        line = line.split("\t")
        
        word1 = line[0]
        word2 = line[1]
        
        similarity = model.similarity(word1, word2) #Word2Vecのコサイン類似度を計算
        
        line.append(str(similarity))
        
        word_list.append(line)
        
human = np.array(word_list).T[2] #すべての行の2列目=転置行列の2行目
w2v = np.array(word_list).T[3] #Word2Vecの類似度の行列

correlation, pvalue = spearmanr(human, w2v) #スピアマンの相関係数

print("スピアマンの相関係数：", correlation)