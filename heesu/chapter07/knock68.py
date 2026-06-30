from knock61 import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np

DATA_DIR = "./SST-2"

train = load_dataset(f"{DATA_DIR}/train.tsv")

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform([d["feature"] for d in train])
y_train = [d["label"] for d in train]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

feature_names = vectorizer.get_feature_names_out()
# coef_[0] corresponds to the positive class (label "1") in binary classification
weights = model.coef_[0]
sorted_indices = np.argsort(weights)

print("Top 20 features with highest weights (strongly Positive):")
for i in sorted_indices[-20:][::-1]:
    print(f"  {feature_names[i]:<20} {weights[i]:+.4f}")

print("\nTop 20 features with lowest weights (strongly Negative):")
for i in sorted_indices[:20]:
    print(f"  {feature_names[i]:<20} {weights[i]:+.4f}")
