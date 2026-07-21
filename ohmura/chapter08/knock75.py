import torch
from torch.nn.utils.rnn import pad_sequence
from knock70 import load_embeddings
from knock71 import create_dataset

def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)
    
    input_ids = [item['input_ids'] for item in sorted_batch]
    
    labels = torch.tensor([[item['label'].item()] for item in sorted_batch], dtype=torch.float32)
    
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    return {'input_ids': padded_ids, 'label': labels}

if __name__ == '__main__':
    _, word2id, _ = load_embeddings(limit=100000)
    train_data = create_dataset('../chapter07/SST-2/train.tsv', word2id)
    
    sample_batch = train_data[:4]
    
    print("【パディング前の各データの単語数】")
    for i, item in enumerate(sample_batch):
        print(f"事例{i}: {len(item['input_ids'])}単語")
        
    collated = collate_fn(sample_batch)
    
    print("\n【collate関数を通した結果】")
    print("input_ids:")
    print(collated['input_ids'])
    print("\nlabel:")
    print(collated['label'])