#与えられたテキストのポジネガを予測するプログラムを実装せよ。例えば、テキストとして”the worst movie I ‘ve ever seen”を与え、ロジスティック回帰モデルの予測結果を確認せよ。

import knock62

text = "the worst movie I ‘ve ever seen"

feature = {} #特徴量を格納する辞書

words = text.split()

for word in words:
    
    if word in feature:
        
        feature[word] += 1
    
    else:
        
        feature[word] = 1   

feature_vector = knock62.vectorizer.transform(feature) #特徴ベクトルを作成

label_pred = knock62.logistic.predict(feature_vector) #ポジネガを予測

if __name__ == "__main__":
    
    if label_pred == "1":
        
        print(text, "：", "positive")
    
    else:
        
        print(text, "：", "negative")