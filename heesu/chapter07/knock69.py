from knock61 import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./SST-2"

train = load_dataset(f"{DATA_DIR}/train.tsv")
dev = load_dataset(f"{DATA_DIR}/dev.tsv")

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform([d["feature"] for d in train])
y_train = [d["label"] for d in train]
X_dev = vectorizer.transform([d["feature"] for d in dev])
y_dev = [d["label"] for d in dev]

C_values = np.logspace(-3, 3, 13)
accuracies = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_dev, model.predict(X_dev))
    accuracies.append(acc)
    print(f"C={C:.5f}  accuracy={acc:.4f}")

plt.figure(figsize=(8, 5))
plt.semilogx(C_values, accuracies, marker="o")
plt.xlabel("Regularization parameter C")
plt.ylabel("Accuracy on dev set")
plt.title("Logistic Regression: Regularization vs Dev Accuracy")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("knock69_plot.png", dpi=150)
print("\nPlot saved to knock69_plot.png")
