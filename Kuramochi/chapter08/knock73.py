from knock70 import Word2Vectors
from knock72 import data_main
from knock72 import SingleLayerNN

import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model, X, y, epochs, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()                      # ２値分類で使われる損失関数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_list  = []
    best_loss  = float('inf')
    best_state = None

    for epoch in range(epochs):
        y_pred = model(X)
        loss   = criterion(y_pred, y)    # y_predを渡すことにより、y_predの計算経路がlossに記憶されている??

        # lossとoptimizerの仕組みは未だブラックボックス。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            loss_list.append(loss.item())

    torch.save({
    "model_state": model.state_dict(),
    "best_loss"  : min(loss_list),
    "epochs"     : len(loss_list),
}, "checkpoint.pth")
    
    return model, loss_list

if __name__=="__main__":

    #各path
    path_embedding = "GoogleNews-vectors-negative300.bin"
    path_train     = "SST-2/train.tsv"
    path_test      = "SST-2/dev.tsv"

    # データの準備
    word2id, id2word, E = Word2Vectors(path_embedding)
    train_X, train_y    = data_main(path_train, word2id, E)

    # モデルの構築
    dim_input  = 300
    dim_output = 1
    model      = SingleLayerNN(dim_input, dim_output)

    # モデルの学習
    epochs = 10000
    lr     = 0.01
    model, loss_list = train_model(model, train_X, train_y, epochs=epochs, lr=lr)

    # モデルの描画
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    # modelの保存
    torch.save(model.state_dict(), "model_73.pth")
    print("学習および保存が完了しました。")