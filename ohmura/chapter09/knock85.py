import pandas as pd
from transformers import BertTokenizer

def load_and_tokenize(file_path, tokenizer):
    """
    TSVファイルを読み込み、テキストをトークン列に変換する関数
    """
    df = pd.read_csv(file_path, sep='\t')
    
    df['tokens'] = df['sentence'].apply(tokenizer.tokenize)
    
    return df

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_path = '../chapter07/SST-2/train.tsv'
    dev_path = '../chapter07/SST-2/dev.tsv'
    
    train_df = load_and_tokenize(train_path, tokenizer)
    dev_df = load_and_tokenize(dev_path, tokenizer)
    
    print("\n=== train.tsv (先頭3件) ===")
    print(train_df[['sentence', 'label', 'tokens']].head(3))
    
    print(f"\n訓練データの件数: {len(train_df)}件")
    print(f"開発データの件数: {len(dev_df)}件")