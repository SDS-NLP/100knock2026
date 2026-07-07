import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from knock70 import load_embeddings
from knock71 import create_dataset
from knock72 import BoWModel
from knock75 import collate_fn

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].squeeze(1).long().to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    embeddings, word2id, _ = load_embeddings(limit=100000)
    train_data = create_dataset('../chapter07/SST-2/train.tsv', word2id)
    dev_data = create_dataset('../chapter07/SST-2/dev.tsv', word2id)
    
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = BoWModel(embeddings)
    model.embedding.weight.requires_grad = True
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    print("ファインチューニングありのミニバッチ学習を開始します...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].squeeze(1).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")
        
    dev_acc = calculate_accuracy(model, dev_loader, device)
    
    print(f"開発データの正解率: {dev_acc:.4f}")