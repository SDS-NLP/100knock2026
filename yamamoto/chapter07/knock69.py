#ロジスティック回帰モデルを学習するとき、正則化の係数（ハイパーパラメータ）を調整することで、学習時の適合度合いを制御できる。正則化の係数を変化させながらロジスティック回帰モデルを学習し、検証データ上の正解率を求めよ。実験の結果は、正則化パラメータを横軸、正解率を縦軸としたグラフにまとめよ。

import knock62
import knock63
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

feature_train = knock62.feature_train
label_train = knock62.label_train

feature_dev = knock63.feature_dev
label_dev = knock63.label_dev

parameters = np.arange(0.1, 2.1, 0.05) #0.1~2.0まで0.05刻みの数値を格納するリストを作成し、正則化の係数とする
accuracy_list = [] #各係数に対する正解率を格納

for parameter in parameters: #各係数に対してモデルを学習、評価
    
    model = LogisticRegression(C = parameter, max_iter = 500) #正則化の係数を引数Cで指定
    
    model.fit(feature_train, label_train)
    
    label_pred = model.predict(feature_dev)
    
    accuracy = accuracy_score(label_dev, label_pred)
    
    accuracy_list.append(accuracy)

if __name__ == "__main__":
    
    plt.subplot()
    plt.scatter(parameters, accuracy_list)
    plt.xlabel("regularization")
    plt.ylabel("accuracy")

    plt.show()