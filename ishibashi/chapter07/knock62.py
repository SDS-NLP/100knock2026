import os
import csv
import joblib
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    X_dicts = []
    y = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)

        for row in reader:
            text, label = row

            X_dicts.append(dict(Counter(text.split(' '))))
            y.append(label)

    return X_dicts, y

def train_and_save_model():
    train_file = './chapter07/SST-2/train.tsv'
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    print("学習データの読み込み中")
    train_X_dicts, train_y = load_data(train_file)

    print("辞書を疎行列に変換中")
    vec = DictVectorizer()
    X_train = vec.fit_transform(train_X_dicts)

    print("ロジスティック回帰モデルを学習中")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_y)

    print("モデルを保存中")
    joblib.dump(clf, model_file)
    joblib.dump(vec, vec_file)

    print(f"保存完了({model_file}, {vec_file})")

if __name__ == "__main__":
    train_and_save_model()