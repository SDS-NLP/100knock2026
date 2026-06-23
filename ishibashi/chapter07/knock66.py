import os
import csv
import joblib
from collections import Counter
from sklearn.metrics import confusion_matrix

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

def generate_matrix():
    dev_file = './chapter07/SST-2/dev.tsv'
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    if not all(os.path.exists(f) for f in [dev_file, model_file, vec_file]):
        print("必要なファイルのいずれかが見つかりません")
        return
    
    clf = joblib.load(model_file)
    vec = joblib.load(vec_file)

    X_dicts, y_true = load_data(dev_file)

    X_dev = vec.transform(X_dicts)
    y_pred = clf.predict(X_dev)

    cm = confusion_matrix(y_true, y_pred, labels=['0', '1'])
    
    print('Predicted Nega(0), Predicted Posi(1)')
    print(f'Actual Negative(0): {cm[0][0]} {cm[0][1]}')
    print(f'Actual Positive(1): {cm[1][0]} {cm[1][1]}')

if __name__ == "__main__":
    generate_matrix()