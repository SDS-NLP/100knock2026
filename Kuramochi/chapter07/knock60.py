import pandas as pd


def count_sst2_labels(file_path):
    # TSVファイルとして読み込み
    df = pd.read_csv(file_path, sep='\t')
    
    # 'label'列の値をカウント
    counts = df['label'].value_counts()
    
    # 結果を見やすく出力
    print(f"【{file_path} の事例数】")
    print(f"ポジティブ (1): {counts.get(1, 0)} 件")
    print(f"ネガティブ (0): {counts.get(0, 0)} 件")
    print("-" * 30)

# train.tsv と dev.tsv の集計を実行
count_sst2_labels('SST-2/train.tsv')
count_sst2_labels('SST-2/dev.tsv')