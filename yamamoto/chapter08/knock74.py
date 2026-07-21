#問題73で学習したモデルの開発セットにおける正解率を求めよ。

import knock70
import knock71
import knock72
import torch

embedding = knock70.embeddings
dev_data = knock71.dev_data

model = knock72.NeuralNetworkModel(embedding) #モデルを呼び出す
model.load_state_dict(torch.load("model.pt")) #保存した重みをファイルからロード
model.eval() #モデルを評価用に切り替える

score = 0

with torch.no_grad(): #勾配を更新しない(学習しない)
    
    for example in dev_data:
    
        input_ids = example["input_ids"].unsqueeze(0)
        label = float(example["label"])
        
        logit = model(input_ids) #モデルで分類スコアを計算
        
        prob = torch.sigmoid(logit) #シグモイド関数で確率に変換(ロジスティック回帰)
        
        pred = (prob >= 0.5).float() #0.5以上で1.0(True), それ以下は0.0(False)でラベルを予測
        
        if pred.item() == label:
            
            score += 1
    
    accuracy = score / len(dev_data)

if __name__ == "__main__":
    
    print("正解率：", accuracy)