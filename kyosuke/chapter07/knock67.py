import pandas as pd
from collections import Counter
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = joblib.load('lr_model.joblib')
vec = joblib.load('vectorizer.joblib')

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    X_dicts = []
    y_true = []
    for _, row in df.iterrows():
        text = str(row['sentence'])
        X_dicts.append(dict(Counter(text.split())))
        y_true.append(int(row['label']))
    return X_dicts, y_true

def evaluate_model(X_dicts, y_true, data_name):
    X = vec.transform(X_dicts)
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n【{data_name}】の評価結果")
    print(f"正解率: {acc:.4f}")
    print(f"適合率: {prec:.4f}")
    print(f"再現率: {rec:.4f}")
    print(f"F1スコア: {f1:.4f}")

X_train_dicts, y_train_true = load_data('SST-2/train.tsv')
evaluate_model(X_train_dicts, y_train_true, "学習データ")

X_dev_dicts, y_dev_true = load_data('SST-2/dev.tsv')
evaluate_model(X_dev_dicts, y_dev_true, "検証データ")