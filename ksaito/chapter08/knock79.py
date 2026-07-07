from pathlib import Path

import torch
from torch import nn

from knock71 import load_embeddings, load_sst2_as_ids
from knock76 import accuracy, train_minibatch
from knock77 import device_for_gpu


class CnnSentimentClassifier(nn.Module):
    def __init__(self, embedding_matrix, channels=128, kernel_sizes=(3, 4, 5)):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=False, padding_idx=0
        )
        dim = weights.size(1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, channels, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            pooled.append(torch.max(features, dim=2).values)
        return self.linear(self.dropout(torch.cat(pooled, dim=1)))


def main():
    device = device_for_gpu()
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    embedding_matrix, token_to_id, _ = load_embeddings(limit=100000)
    train, dev = load_sst2_as_ids(token_to_id)
    model = CnnSentimentClassifier(embedding_matrix)
    train_minibatch(model, train, dev, device=device, epochs=5, batch_size=256, lr=5e-4)
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/knock79_cnn.pt")
    print(f"final_dev_accuracy: {accuracy(model, dev, device=device):.4f}")


if __name__ == "__main__":
    main()
