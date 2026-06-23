#学習したロジスティック回帰モデルの中で、重みの高い特徴量トップ20と、重みの低い特徴量トップ20を確認せよ。

import knock62

names = knock62.vectorizer.get_feature_names_out() #特徴ベクトルの名前
weights = knock62.logistic.coef_ #ロジスティック回帰モデルの回帰係数

features = {} #特徴量をkey, 回帰係数をvalueとする辞書

for i in range(len(names)):
    
    features[names[i]] = weights[0][i] #回帰係数はweightsの各列に入っている

sorted_features = sorted(features.items(), key = lambda x: x[1]) #valueについて昇順に並び替え
reversed_features = sorted(features.items(), key = lambda x: x[1], reverse = True) #valueについて降順に並び替え

if __name__ == "__main__":
    
    for key, value in reversed_features[:20]:
        
        print(key, value)
        
    print("...")    
    
    for key, value in sorted_features[19::-1]:
        
        print(key, value)