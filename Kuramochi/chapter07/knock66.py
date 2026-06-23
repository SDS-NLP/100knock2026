from knock61 import process_tsv_to_bow
import joblib

from sklearn.metrics import confusion_matrix

# 学習済みモデルとベクトライザーの読み込み
loaded_model      = joblib.load('sst2_lr_model.pkl')
loaded_vectorizer = joblib.load('sst2_vectorizer.pkl')

# テストデータの用意
train_data = process_tsv_to_bow('SST-2/train.tsv')
test_data  = process_tsv_to_bow('SST-2/dev.tsv')

# BoW特徴量のベクトル化
X_train_matrix = loaded_vectorizer.transform([data['feature'] for data in train_data])
X_test_matrix  = loaded_vectorizer.transform([data['feature'] for data in test_data])
Y_train        = [int(data['label']) for data in train_data]
Y_test         = [int(data['label']) for data in test_data]


Y_test_pred = loaded_model.predict(X_test_matrix)

cm = confusion_matrix(Y_test, Y_test_pred)
print(cm)