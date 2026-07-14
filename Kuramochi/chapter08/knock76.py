import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from knock70 import Word2Vectors
from knock75 import process_tsv_to_dict, collate


def batches2data(batches, E):
    X_list = []
    y_list = []

    input_ids = batches["input_ids"]
    labels    = batches["label"]

    for i in range(len(input_ids)):
        ids            = input_ids[i]
        average_vector = E[ids].mean(dim=0)
        label          = labels[i]

        X_list.append(average_vector)
        y_list.append(label)

    X = torch.stack(X_list)
    y = torch.stack(y_list)
    return X, y


class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train_model_minibatch(model, X, y, epochs=100, batch_size=32, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


if __name__=="__main__":
    file_path_embedding = "GoogleNews-vectors-negative300.bin"
    file_path_train     = "SST-2/train.tsv"
    file_path_test      = "SST-2/dev.tsv"

    word2id, id2word, E = Word2Vectors(file_path_embedding)

    train_dataset = process_tsv_to_dict(file_path_train, word2id)
    test_dataset  = process_tsv_to_dict(file_path_test, word2id)

    train_batches = collate(train_dataset)
    test_batches  = collate(test_dataset)

    X_train, y_train = batches2data(train_batches, E)
    X_test, y_test   = batches2data(test_batches, E)

    model = SingleLayerNN(300, 1)
    model = train_model_minibatch(model, X_train, y_train, epochs=100, batch_size=32, lr=0.01)

    with torch.no_grad():
        test_pred = torch.sigmoid(model(X_test))
        test_pred = (test_pred >= 0.5).float()
        accuracy  = (test_pred == y_test).float().mean().item()

    print(f"\nTest accuracy: {accuracy:.4f}")
