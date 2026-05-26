import os
import urllib.request
import zipfile
import csv
import gensim.downloader as api
from scipy.stats import spearmanr
from tqdm import tqdm


url = "http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip"
zip_path = "wordsim353.zip"
extract_dir = "wordsim353_data"

if not os.path.exists(zip_path):
    print("WordSimilarity-353データセットをダウンロードしています")
    urllib.request.urlretrieve(url, zip_path)

if not os.path.exists(extract_dir):
    print("ZIPファイルを解凍しています")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

print("Word2Vecモデルを読み込んでいます")
model = api.load("word2vec-google-news-300")


# combined.csv が人間による評価データ
csv_file = os.path.join(extract_dir, "combined.csv")

human_scores = []
w2v_scores = []
skipped_count = 0

print("単語ペアの類似度を計算中")
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 1行目（ヘッダー: Word 1, Word 2, Human (mean)）をスキップ
    
    for row in reader:
        word1, word2 = row[0], row[1]
        human_score = float(row[2])
        
        # 両方の単語がモデルの語彙（辞書）に存在するかチェック
        if word1 in model and word2 in model:
            w2v_score = model.similarity(word1, word2)
            
            human_scores.append(human_score)
            w2v_scores.append(w2v_score)
        else:
            skipped_count += 1


# spearmanrは、(相関係数, p値) のタプルを返します
correlation, pvalue = spearmanr(human_scores, w2v_scores)

print("\n--- 実行結果 ---")
print(f"評価した単語ペア数: {len(human_scores)} 件 (未知語でスキップ: {skipped_count} 件)")
print(f"スピアマン相関係数: {correlation:.4f}")

#評価した単語ペア数: 353 件 (未知語でスキップ: 0 件)
# スピアマン相関係数: 0.7000