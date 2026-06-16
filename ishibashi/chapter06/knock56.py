import csv
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

def evaluate_wordsim353():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'
    wordsim_file = './chapter06/wordsim353/combined.csv' 

    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=500000)

    human_scores = []
    model_scores = []

    with open(wordsim_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) != 3:
                continue
            
            word1, word2, human_score_str = row
        
            if word1 in model and word2 in model:
                model_score = model.similarity(word1, word2)
                human_score = float(human_score_str)
                
                model_scores.append(model_score)
                human_scores.append(human_score)

    if len(human_scores) > 0:
        correlation, pvalue = spearmanr(human_scores, model_scores)
        print(f"評価した単語ペア数: {len(human_scores)} 件")
        print(f"スピアマンの順位相関係数: {correlation:.4f}")
    else:
        print("評価可能な単語ペアが見つかりませんでした。")

if __name__ == "__main__":
    evaluate_wordsim353()