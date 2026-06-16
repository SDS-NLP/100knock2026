import os
import csv
import joblib
from collections import Counter

def predict_first_instance():
    dev_file = './chapter07/SST-2/dev.tsv'
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    if not all(os.path.exists(f) for f in [dev_file, model_file, vec_file]):
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

    pred_label = clf.predict(X_first)[0]

    print(f'対象テキスト: "{text}"')
    print(f'正解ラベル: "{true_label}" (型: {type(true_label).__name__})')
    print(f'予測ラベル: "{pred_label}" (型: {type(pred_label).__name__})')

    is_match = str(true_label) == str(pred_label)
    print(f'判定: {"一致" if is_match else "不一致"}')

if __name__ == "__main__":
    predict_first_instance()