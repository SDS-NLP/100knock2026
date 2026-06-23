import pandas as pd
from collections import Counter
import joblib
from sklearn.metrics import confusion_matrix

model = joblib.load('lr_model.joblib')
vec = joblib.load('vectorizer.joblib')

def make_list(file_path):
    dataset = []
    df = pd.read_csv(file_path, sep='\t')
    for _, row in df.iterrows():
        text = row["sentence"]
        label = row["label"]
        feature = dict(Counter(text.split()))
        instance = {
            "text": text,
            "label": label,
            "feature": feature
        }
        dataset.append(instance)
    return dataset

dev_data = make_list('SST-2/dev.tsv')

X_dev_dicts = [data['feature'] for data in dev_data]
y_dev_true = [data['label'] for data in dev_data]
X_dev = vec.transform(X_dev_dicts)

y_dev_pred = model.predict(X_dev)

cm = confusion_matrix(y_dev_true, y_dev_pred)
print(cm)

print(f"真陰性: {cm[0][0]} 件")
print(f"偽陽性: {cm[0][1]} 件")
print(f"偽陰性: {cm[1][0]} 件")
print(f"真陽性: {cm[1][1]} 件")