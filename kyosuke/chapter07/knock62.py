import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

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

train_data = make_list("SST-2/train.tsv")
vec = DictVectorizer()

x_train = vec.fit_transform([instance["feature"] for instance in train_data])
y_train = [instance["label"] for instance in train_data]

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)

joblib.dump(model, 'lr_model.joblib')
joblib.dump(vec, 'vectorizer.joblib')