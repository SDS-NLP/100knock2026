#学習したロジスティック回帰モデルを用い、検証データの先頭の事例のラベル（ポジネガ）を予測せよ。また、予測されたラベルが検証データで付与されていたラベルと一致しているか、確認せよ。

import knock61
import knock62
from sklearn.metrics import accuracy_score

bow_dev = knock61.bow_dev

feature = [] #検証データの特徴量を格納
label_dev = [] #検証データのラベルを格納

for i in range(len(bow_dev)):
    
    feature.append(bow_dev[i]["feature"])
    label_dev.append(bow_dev[i]["label"])

feature_dev = knock62.vectorizer.transform(feature) #特徴量の辞書を数値ベクトルに変換
label_pred = knock62.logistic.predict(feature_dev) #ロジスティック回帰モデルで検証データの特徴ベクトルのラベルを予測

accuracy = accuracy_score(label_dev, label_pred) #予測ラベルの正解率を計算

if __name__ == "__main__":
    
    print("正解率", accuracy)