from knock70 import Word2Vectors
from knock71 import process_tsv_to_dict
import torch
import torch.nn as nn

def ids2average(ids, E):
    return E[ids].mean(dim=0)

def dict2data(dicts: dict, E):

    X_list = []
    y_list = []

    for d in dicts:

        ids            = d["input_ids"]
        average_vector = ids2average(ids, E)
        label          = d["label"]

        X_list.append(average_vector)
        y_list.append(label)
    
    X = torch.stack(X_list)
    y = torch.stack(y_list)

    return X, y

def data_main(path_data, word2id, E):
    dicts = process_tsv_to_dict(path_data, word2id)
    X, y  = dict2data(dicts, E)
    return X, y


class SingleLayerNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__=="__main__":

    # モデルの構築
    d_in  = 300
    d_out = 1
    model = SingleLayerNN(d_in, d_out)
    print(model)

    # データの準備
    file_path_embedding = "GoogleNews-vectors-negative300.bin"
    file_path_train     = "SST-2/train.tsv"
    file_path_test      = "SST-2/dev.tsv"

    word2id, id2word, E = Word2Vectors(file_path_embedding)
    train_X, train_y = data_main(file_path_train, word2id, E)
    test_X, test_y   = data_main(file_path_test, word2id, E)
    print(train_X[:5])
    print(train_y[:5])