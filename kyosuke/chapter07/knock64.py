import joblib
import pandas as pd
from collections import Counter

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

first_data = make_list("SST-2/train.tsv")[0]
feature = first_data["feature"]
X_dev_first = vec.transform([feature])
prob = model.predict_proba(X_dev_first)[0]

prob_negative = prob[0]
prob_positive = prob[1]

print(f'Negative: {prob_negative:.4f}, Positive: {prob_positive:.4f}')