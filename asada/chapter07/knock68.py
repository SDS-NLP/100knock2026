import polars as pl
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from knock61 import objectify

train_data = objectify("SST-2/train.tsv")

model_logistic = LogisticRegression(max_iter=1000)

features = [item["feature"] for item in train_data]
y = [item["label"] for item in train_data]

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(features)
model_logistic.fit(X, y)

feature_names = vec.get_feature_names_out()
weights = model_logistic.coef_[0]

weight_df = pl.DataFrame({"word": feature_names, "weight": weights})

print(
    f"重みの高い特徴量トップ20:\n{weight_df.sort('weight', descending=True).head(20)}"
)

print(f"重みの低い特徴量トップ20:\n{weight_df.sort('weight').head(20)}")
