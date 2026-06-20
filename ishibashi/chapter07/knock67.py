import os
import csv
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    X_dicts = []
    y = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)

        for row in reader:
            if len(row) != 2:
                continue
            text, label = row
            X_dicts.append(dict(Counter(text.split(' '))))
            y.append(label)

    return X_dicts, y

def evaluate_perfomance(clf, vec, file_path, data_name):
    X_dicts, y_true = load_data(file_path)
    X_features = vec.transform(X_dicts)
    y_pred = clf.predict(X_features)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='1', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='1', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='1', zero_division=0)

    print(f"正解率(Accuracy): {acc:.4f}")
    print(f"適合率(Precision): {prec:.4f}")
    print(f"再現率(Recall): {rec:.4f}")
    print(f"F1スコア(F1-score): {f1:.4f}")

    return acc, f1

def main():
    train_file = './chapter07/SST-2/train.tsv'
    dev_file = './chapter07/SST-2/dev.tsv'
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    if not all(os.path.exists(f) for f in [train_file, dev_file,model_file, vec_file]):
        print("必要なファイルのいずれかが見つかりません")
        return
    
    clf = joblib.load(model_file)
    vec = joblib.load(vec_file)

    evaluate_perfomance(clf, vec, train_file, "学習データ(train.tsv)")
    evaluate_perfomance(clf, vec, dev_file, "検証データ(dev.tsv)")

if __name__ == "__main__":
    main()