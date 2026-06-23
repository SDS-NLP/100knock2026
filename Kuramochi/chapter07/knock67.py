from knock61 import process_tsv_to_bow
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

Y_train_pred    = loaded_model.predict(X_train_matrix)
Y_test_pred     = loaded_model.predict(X_test_matrix)

all_labels      = list(Y_train) + list(Y_test)
all_predictions = list(Y_train_pred) + list(Y_test_pred)

print("===score on the train data===")
print(f"Accuracy : {accuracy_score(Y_train, Y_train_pred)}")
print(f"Precision: {precision_score(Y_train, Y_train_pred)}")
print(f"Recall   : {recall_score(Y_train, Y_train_pred)}")
print(f"F1       : {f1_score(Y_train, Y_train_pred)}\n")

print("===score on the test data===")
print(f"Accuracy : {accuracy_score(Y_test, Y_test_pred)}")
print(f"Precision: {precision_score(Y_test, Y_test_pred)}")
print(f"Recall   : {recall_score(Y_test, Y_test_pred)}")
print(f"F1       : {f1_score(Y_test, Y_test_pred)}\n")

print("===score on the all data===")
print(f"Accuracy : {accuracy_score(all_labels, all_predictions)}")
print(f"Precision: {precision_score(all_labels, all_predictions)}")
print(f"Recall   : {recall_score(all_labels, all_predictions)}")
print(f"F1       : {f1_score(all_labels, all_predictions)}\n")