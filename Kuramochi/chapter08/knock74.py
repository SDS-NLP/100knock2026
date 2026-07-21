from knock70 import Word2Vectors
from knock72 import data_main
from knock72 import SingleLayerNN

import torch
import torch.nn as nn

"[Epoch [10000/10000], Loss: 0.4946]のモデルを使っています。"

def evaluate_model(model, X, y):

    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
        logits = model(X)
        loss   = criterion(logits, y)

        # 0.5以上だったら1を返す（.floatでTrue/False→1/0）
        preds    = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (preds == y).float().mean()

    return loss.item(), accuracy.item()

if __name__=="__main__":

     #各path
    path_embedding = "GoogleNews-vectors-negative300.bin"
    path_train     = "SST-2/train.tsv"
    path_test      = "SST-2/dev.tsv"

    # データの準備
    word2id, id2word, E = Word2Vectors(path_embedding)
    train_X, train_y    = data_main(path_train, word2id, E)
    test_X, test_y      = data_main(path_test, word2id, E)

    # モデルの構築・読み込み
    dim_input  = 300
    dim_output = 1
    model      = SingleLayerNN(dim_input, dim_output)

    checkpoint = torch.load("checkpoint.pth")
    model.load_state_dict(checkpoint["model_state"])
    print(f"train_loss: {checkpoint['best_loss']}, train_epochs: {checkpoint['epochs']}")

    # 評価
    loss_train, accuracy_train = evaluate_model(model, train_X, train_y)
    loss_test, accuracy_test   = evaluate_model(model, test_X, test_y)
    
    print(f"Train Loss: {loss_train:.4f}, Train Accuracy: {accuracy_train:.4f}")
    print(f"Val Loss  : {loss_test:.4f}, Val Accuracy  : {accuracy_test:.4f}")


    