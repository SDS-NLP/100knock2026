#学習したロジスティック回帰モデルの正解率、適合率、再現率、F1スコアを、学習データおよび検証データ上で計測せよ。

import knock62
import knock63
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

feature_train = knock62.feature_train #特徴ベクトルの学習データ
label_train = knock62.label_train #ラベルの学習データ
label_pred_train = knock62.logistic.predict(feature_train) #学習データを用いた予測ラベル

label_dev = knock63.label_dev #ラベルの検証データ
label_pred_dev = knock63.label_pred #検証データを用いた予測ラベル

accuracy_train = accuracy_score(label_train, label_pred_train) #正解率
precision_train = precision_score(label_train, label_pred_train, pos_label = "1") #適合率
recall_train = recall_score(label_train, label_pred_train, pos_label = "1") #再現率
f1_train = f1_score(label_train, label_pred_train, pos_label = "1") #F1スコア

accuracy_dev = accuracy_score(label_dev, label_pred_dev)
precision_dev = precision_score(label_dev, label_pred_dev, pos_label = "1")
recall_dev = recall_score(label_dev, label_pred_dev, pos_label = "1")
f1_dev = f1_score(label_dev, label_pred_dev, pos_label = "1")

if __name__ == "__main__":
    
    print("正解率：", "学習データ", accuracy_train, "検証データ", accuracy_dev)
    print("適合率：", "学習データ", precision_train, "検証データ", precision_dev)
    print("再現率：", "学習データ", recall_train, "検証データ", recall_dev)
    print("F1スコア：", "学習データ", f1_train, "検証データ", f1_dev)