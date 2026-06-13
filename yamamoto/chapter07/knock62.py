#61で構築した学習データの特徴ベクトルを用いて、ロジスティック回帰モデルを学習せよ。

import knock61
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

bow_train = knock61.bow_train
feature = [] #訓練データの特徴ベクトルだけを格納
label = [] #訓練データのラベルだけを格納

for i in range(len(bow_train)):
    
    feature.append(bow_train[i]["feature"])
    label.append(bow_train[i]["label"])

vectorizer = DictVectorizer()

feature_train = vectorizer.fit_transform(feature) #特徴量の辞書に対して単語の一覧を作ってから数値ベクトルに変換
label_train = label

logistic = LogisticRegression(max_iter = 1000) #max_iterで学習の反復回数を指定(デフォルトは100回で、今回は収束しなかった)

logistic.fit(feature_train, label_train) #特徴ベクトルからラベルを予測するようにロジスティック回帰モデルを学習