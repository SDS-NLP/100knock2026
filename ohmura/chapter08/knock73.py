import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from knock70 import load_embeddings
from knock71 import create_dataset
from knock72 import BoWModel

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.tensor([item['label'].item() for item in batch], dtype=torch.long)
    
    padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    return {'input_ids': padded_ids, 'labels': labels}

if __name__ == '__main__':
    embeddings, word2id, _ = load_embeddings(limit=100000)
    train_data = create_dataset('../chapter07/SST-2/train.tsv', word2id)
    
    batch_size = 64
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = BoWModel(embeddings)
    model.embedding.weight.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            inputs = batch['input_ids']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")