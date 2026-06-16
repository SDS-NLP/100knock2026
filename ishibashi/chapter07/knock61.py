import os
import csv
from collections import Counter

def load_and_extract_features(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"フィアルが見つかりません: {file_path}")
    
    dataset = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        
        for line_num, row in enumerate(reader, start=2):
            text = row[0]
            label = row[1]

            tokens = text.split(' ')
            feature = dict(Counter(tokens))

            instance = {
                'text': text,
                'label': label,
                'feature': feature
            }
            dataset.append(instance)

    return dataset

def main():
    train_file = './chapter07/SST-2/train.tsv'
    dev_file = './chapter07/SST-2/dev.tsv'

    train_data = load_and_extract_features(train_file)
    dev_data = load_and_extract_features(dev_file)

    print("学習データの最初の事例")
    first_instance_train = train_data[0]
    for key, value in first_instance_train.items():
        print(f"{key}: {value}")
    
    print("検証データの最初の事例")
    first_instance_dev = dev_data[0]
    for key, value in first_instance_dev.items():
        print(f"{key}, {value}")

if __name__ == "__main__":
    main()