import csv
from collections import Counter

# テキストデータを受け取って、BoW特長料を辞書型で返す関数
def text_to_bow(text)-> dict:

    tokens       = text.split()
    count_tokens = Counter(tokens)

    return dict(count_tokens)

def process_tsv_to_bow(file_path) -> list:

    dataset = []

    # 辞書形式に変換
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t') # https://pythonaiclarifydoubts.com/csv-dictreader/
        
        for row in reader:
            
            # 'sentence'列と'label'列の値を取得
            text  = row.get('sentence', '') 
            label = row.get('label', '')
            
            # BoW特徴量の作成
            tokens         = text.split()
            count_tokens   = Counter(tokens)
            feature_vector = dict(count_tokens)
            
            # 指定されたフォーマットの辞書オブジェクトを作成
            instance = {
                'text'   : text,
                'label'  : label,
                'feature': feature_vector
            }
            
            dataset.append(instance)
            
    return dataset

if __name__ == "__main__":
    # 1. 学習データと検証データを辞書オブジェクトのリストに変換
    train_data = process_tsv_to_bow('SST-2/train.tsv')
    dev_data   = process_tsv_to_bow('SST-2/dev.tsv')

    # 2. 学習データの最初の事例について、正しく変換できたか目視確認
    if train_data:
        first_instance = train_data
        print(first_instance)
