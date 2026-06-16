from knock61 import process_tsv_to_bow
import joblib

# 学習済みモデルとベクトライザーの読み込み
loaded_model      = joblib.load('sst2_lr_model.pkl')
loaded_vectorizer = joblib.load('sst2_vectorizer.pkl')

# テストデータの用意
test_data  = process_tsv_to_bow('SST-2/dev.tsv')

# BoW特徴量のベクトル化
X_test_matrix  = loaded_vectorizer.transform([data['feature'] for data in test_data])
Y_test         = [int(data['label']) for data in test_data]

# 予測の実行
predictions     = loaded_model.predict(X_test_matrix)

# 予測結果の表示
text            = test_data[0]['text']
actual_label    = Y_test[0]
predicted_label = predictions[0]
print(f"テキスト   : {text}")
print(f"正解ラベル : {actual_label}")
print(f"予測ラベル : {predicted_label}")
