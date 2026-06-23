from knock61 import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix

DATA_DIR = "./SST-2"

train = load_dataset(f"{DATA_DIR}/train.tsv")
dev = load_dataset(f"{DATA_DIR}/dev.tsv")

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform([d["feature"] for d in train])
y_train = [d["label"] for d in train]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

X_dev = vectorizer.transform([d["feature"] for d in dev])
y_dev = [d["label"] for d in dev]
y_pred = model.predict(X_dev)

cm = confusion_matrix(y_dev, y_pred, labels=model.classes_)
print("Confusion Matrix (rows=actual, cols=predicted):")
print(f"Labels: {list(model.classes_)}")
print(cm)
print()
print(f"{'':>12}  Predicted 0  Predicted 1")
print(f"Actual 0  {cm[0][0]:>11}  {cm[0][1]:>11}")
print(f"Actual 1  {cm[1][0]:>11}  {cm[1][1]:>11}")
