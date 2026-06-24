import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from knock70 import load_embeddings
from knock71 import create_dataset
from knock72 import BoWModel
from knock73 import collate_fn

def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids']
            labels = batch['labels']
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

if __name__ == '__main__':
    embeddings, word2id, _ = load_embeddings(limit=100000)
    train_data = create_dataset('../chapter07/SST-2/train.tsv', word2id)
    dev_data = create_dataset('../chapter07/SST-2/dev.tsv', word2id)
    
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = BoWModel(embeddings)
    model.embedding.weight.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch['input_ids']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    train_acc = calculate_accuracy(model, train_loader)
    dev_acc = calculate_accuracy(model, dev_loader)
    
    print(f"学習データの正解率: {train_acc:.4f}")
    print(f"開発データの正解率: {dev_acc:.4f}")