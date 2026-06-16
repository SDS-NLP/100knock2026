from knock61 import process_tsv_to_bow
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import joblib

# 大元のデータ
train_data  = process_tsv_to_bow('SST-2/train.tsv')

# X, Y
X_train_dicts = [data['feature']    for data in train_data]
Y_train       = [int(data['label']) for data in train_data]

# BoW特徴量のベクトル化
vectorizer     = DictVectorizer(sparse=True)
X_train_matrix = vectorizer.fit_transform(X_train_dicts)

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train_matrix, Y_train)

# 学習済みモデルとベクトライザーの保存
joblib.dump(model,      'sst2_lr_model.pkl')
joblib.dump(vectorizer, 'sst2_vectorizer.pkl')

print("学習・保存完了")

