import torch
import torch.nn as nn
from knock70 import load_embeddings
from knock71 import create_dataset

class BoWModel(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        self.linear = nn.Linear(embeddings.size(1), 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    embeddings, word2id, _ = load_embeddings(limit=100000)
    train_data = create_dataset('../chapter07/SST-2/train.tsv', word2id)
    
    model = BoWModel(embeddings)
    
    sample_input = train_data[0]['input_ids'].unsqueeze(0)
    
    output = model(sample_input)
    probs = torch.softmax(output, dim=-1)
    
    print(f"入力ID列の形状: {sample_input.shape}")
    print(f"モデルの出力（ロジット）: {output.detach().numpy()}")
    print(f"予測確率: {probs.detach().numpy()}")