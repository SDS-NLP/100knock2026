import pandas as pd
import torch
from knock70 import load_embeddings

def create_dataset(file_path, word2id):
    df = pd.read_csv(file_path, sep='\t')
    dataset = []
    
    for _, row in df.iterrows():
        text = str(row['sentence'])
        label = float(row['label'])
        
        words = text.split()
        input_ids = [word2id[w] for w in words if w in word2id]
        
        if len(input_ids) > 0:
            dataset.append({
                'text': text,
                'label': torch.tensor([label], dtype=torch.float32),
                'input_ids': torch.tensor(input_ids, dtype=torch.long)
            })
            
    return dataset

if __name__ == '__main__':
    _, word2id, _ = load_embeddings(limit=100000)
    
    train_data = create_dataset('../chapter07/SST-2/train.tsv', word2id)
    dev_data = create_dataset('../chapter07/SST-2/dev.tsv', word2id)
    
    print(f"訓練データの事例数: {len(train_data)}")
    print(f"開発データの事例数: {len(dev_data)}")
    print("\n【訓練データの最初の事例】")
    print(train_data[0])