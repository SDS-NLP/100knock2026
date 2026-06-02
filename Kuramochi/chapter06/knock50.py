from gensim.models import KeyedVectors

file_path = 'GoogleNews-vectors-negative300.bin'

print("巨大なモデルを読み込んでいます...")
print("※数分かかることがあります。画面が止まったように見えても、そのままお待ちください🍵")

model = KeyedVectors.load_word2vec_format(file_path, binary=True)

print("読み込みが完了しました！")

# 3. 課題の指示通り、内部的な表記 "United_States" でベクトルを取得します
target_word = 'United_States'
vector      = model[target_word]

# 4. 結果を表示します
print(f"\n--- 単語 '{target_word}' のベクトル情報 ---")
print(f"ベクトルの次元数（数字の個数）: {vector.shape[0]} 次元")
print("ベクトルの内容（最初の10要素のみ表示）:")
print(vector[:10])  # 300個全部出ると長いので、最初の10個をチラ見せします
print("...")