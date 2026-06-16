#学習したロジスティック回帰モデルを用い、検証データの先頭の事例を各ラベル（ポジネガ）に分類するときの条件付き確率を求めよ。

import knock62
import knock63

feature = knock63.feature_dev[0] #検証データにおける最初の事例の特徴ベクトル

prob = knock62.logistic.predict_proba(feature) #ポジネガ予測の条件付き確率(テキストが与えられたとき、ポジネガそれぞれに分類される条件付き確率)

if __name__ == "__main__":
    
    print("ラベル[0, 1]に分類される条件付き確率：", prob)