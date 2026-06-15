from datasets import load_dataset
from collections import Counter

dataset = load_dataset('glue', 'sst2')

for split in ['train', 'validation']:
    counts = Counter(dataset[split]['label'])
    total = counts[0] + counts[1]
    print(f'{split}: positive={counts[1]}, negative={counts[0]}, total={total}')
