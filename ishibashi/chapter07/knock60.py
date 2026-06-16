import os
import csv
from collections import Counter

def count_labels(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)

        for line_num, row in enumerate(reader, start=2):
            label = row[1]
            counts[label] += 1

    return counts

def main():
    train_file = './chapter07/SST-2/train.tsv'
    dev_file = './chapter07/SST-2/dev.tsv'

    train_counts = count_labels(train_file)
    print(f"train.csv\nポジティブ(1): {train_counts.get('1', 0)}件, ネガティブ(0): {train_counts.get('0', 0)}件")

    dev_counts = count_labels(dev_file)
    print(f"dev.csv\nポジティブ(1): {dev_counts.get('1', 0)}件, ネガティブ(0): {dev_counts.get('0', 0)}件")

if __name__ == "__main__":
    main()