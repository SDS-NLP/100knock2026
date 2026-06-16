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

probabilities = loaded_model.predict_proba(X_test_matrix)

# probabilities の中身は [[ネガティブ(0)の確率, ポジティブ(1)の確率]] の形になります
prob_neg = probabilities[0][0]
prob_pos = probabilities[0][1]

print(f"テキスト: {test_data[0]['text']}")
print(f"ネガティブ (0) である確率 : {prob_neg:.4f} ({prob_neg * 100:.2f}%)")
print(f"ポジティブ (1) である確率 : {prob_pos:.4f} ({prob_pos * 100:.2f}%)")