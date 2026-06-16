from datasets import load_dataset
from collections import Counter

dataset = load_dataset('glue', 'sst2')

def to_bow(text):
    return dict(Counter(text.lower().split()))

def make_examples(split):
    return [
        {'text': ex['sentence'], 'label': ex['label'], 'features': to_bow(ex['sentence'])}
        for ex in dataset[split]
    ]

train = make_examples('train')
dev = make_examples('validation')

print(f'train size: {len(train)}')
print(f'dev size:   {len(dev)}')
print(f'\nexample: {train[0]}')
