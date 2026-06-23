from gensim.models import KeyedVectors
import csv
from scipy.stats import spearmanr

# 単語ベクトル空間の読み込み
file_path = 'GoogleNews-vectors-negative300.bin'
print("巨大なモデルを読み込んでいます...")
print("※数分かかることがあります。画面が止まったように見えても、そのままお待ちください🍵")
model = KeyedVectors.load_word2vec_format(file_path, binary=True)
print("モデルの読み込みが完了しました！")

# 

# WordSim353の評価データファイルのパス
ws353_path = 'wordsim353/combined.tab'

# 人間のスコアとモデルのスコアを格納するリスト
human_scores = []
model_scores = []

# 1. データの読み込みと類似度計算
with open(ws353_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    
    for row in reader:
        if len(row) < 3:
            continue
            
        word1 = row[0]
        word2 = row[1]
        human_score = float(row[2])
        
        # モデルの語彙（ボキャブラリー）に両方の単語が存在するかチェック
        if word1 in model and word2 in model:
            model_score = model.similarity(word1, word2)
            human_scores.append(human_score)
            model_scores.append(model_score)

# 2. スピアマンの順位相関係数を計算
correlation, pvalue = spearmanr(human_scores, model_scores)

print(f"評価に使用した単語ペア数: {len(human_scores)} 件")
print(f"スピアマン相関係数: {correlation:.4f}")