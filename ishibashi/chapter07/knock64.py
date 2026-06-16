import os
import csv
import joblib
from collections import Counter

def calc_conditional_probability():
    dev_file = './chapter07/SST-2/dev.tsv'
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    if not all (os.path.exists(f) for f in [dev_file, model_file, vec_file]):
        print("必要なファイルのいずれかが見つかりません")
        return
    
    clf = joblib.load(model_file)
    vec = joblib.load(vec_file)

    with open(dev_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        first_row = next(reader, None)

        text, true_label = first_row

        feature_dict = dict(Counter(text.split(' ')))
        X_first = vec.transform([feature_dict])

        probabilities = clf.predict_proba(X_first)[0]
        classes = clf.classes_

        print(f'対象テキスト: "{text}"')
        print(f'正解ラベル: "{true_label}"')

        for cls, prob in zip(classes, probabilities):
            print(f'確率 P(y="{cls}"|x) = {prob:.4f} ({prob * 100:.2f}%))')

if __name__ == "__main__":
    calc_conditional_probability()