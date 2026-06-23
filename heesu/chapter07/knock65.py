from knock61 import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from collections import Counter

DATA_DIR = "./SST-2"

train = load_dataset(f"{DATA_DIR}/train.tsv")

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform([d["feature"] for d in train])
y_train = [d["label"] for d in train]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


def predict(text):
    feature = dict(Counter(text.split()))
    X = vectorizer.transform([feature])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return label, dict(zip(model.classes_, proba))


text = "the worst movie I 've ever seen"
label, proba = predict(text)
sentiment = "Positive" if label == "1" else "Negative"
print(f"Text: {text}")
print(f"Predicted: {label} ({sentiment})")
for cls, p in sorted(proba.items()):
    print(f"  P(label={cls}) = {p:.4f}")
