#学習したロジスティック回帰モデルの検証データにおける混同行列（confusion matrix）を求めよ。

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import knock63
import matplotlib.pyplot as plt

matrix = confusion_matrix(knock63.label_dev, knock63.label_pred) #混同行列を作成
print(matrix)

if __name__ == "__main__":
    
    display = ConfusionMatrixDisplay(confusion_matrix = matrix) #混同行列を図で描画
    display.plot()

    plt.show()