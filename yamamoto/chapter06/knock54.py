#単語アナロジーの評価データをダウンロードし、国と首都に関する事例（: capital-common-countriesセクション）に対して、vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し、そのベクトルと類似度が最も高い単語と、その類似度を求めよ。求めた単語と類似度は、各事例と一緒に記録せよ。

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

result = []

with open("questions-words.txt", "r", encoding = "utf-8") as file:
    
    capital_flag = False #デフォルトはFalse
    
    for line in file:
        
        line = line.strip()
        
        if line.startswith(":"): #セクション名が書かれている行かどうかを判断
            
            if line == ": capital-common-countries": #目的のセクションのとき
                
                capital_flag = True
                
                continue #セクション名の行自体は処理せず飛ばす
            
            if capital_flag == True: #目的のセクションにいた状態(True)で別のセクションの開始(":")を見つけたので終了
                
                break
            
            continue #まだ目的のセクションに到達していないときは飛ばす
        
        if capital_flag == False: #まだ目的のセクションに到達していないときの単語の行は飛ばす
            
            continue
        
        word1, word2, word3, word4 = line.split()
        
        if word1 not in model or word2 not in model or word3 not in model or word4 not in model:
            
            continue
        
        vector = model[word2] - model[word1] + model[word3]
        
        list = model.most_similar(vector, topn = 1)
        
        pred_word, similarity = list[0]
    
        result.append([word1, word2, word3, pred_word, similarity])

if __name__ == "__main__":
    
    print(len(result))
    print(result[:10])