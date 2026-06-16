from gensim.models import KeyedVectors

print("巨大なモデルを読み込んでいます...")
file_path = 'GoogleNews-vectors-negative300.bin'
model     = KeyedVectors.load_word2vec_format(file_path, binary=True)
print('読み込みが完了しました.')

input_file  = 'questions-words.txt'
output_file = 'knock54_out.txt'

target_category = ': capital-common-countries'
is_target       = False

print("単語アナロジーの実験を開始します（数十秒〜1分ほどかかります）...")

# ファイルを読み込み用(r)と書き込み用(w)で同時に開く
with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        line = line.strip() # 行末の改行などを削除
        
        # 1. セクション（カテゴリ）の切り替わりを検知
        if line.startswith(':'):
            is_target = (line == target_category)
            continue
            
        # ※is_target が True のときだけ、以降の処理を実行するようにします。
        # 2. 目的のセクションの中身だった場合のみ計算を実行
        if is_target:
            words = line.split()
            
            if len(words) == 4:
                w1, w2, w3, w4 = words
                
                try:
                    # 3. アナロジー計算：vec(w2) - vec(w1) + vec(w3)
                    # result は [(予測単語, 類似度), ...] という形式のリストで返ってきてその中でもっとも類似度の高いものを1件だけ取得します。
                    result         = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)
                    predicted_word = result[0][0]
                    similarity     = result[0][1]
                    
                    # 4. 元のデータに「予測単語」と「類似度」をくっつけてファイルに書き込む
                    f_out.write(f"{line} {predicted_word} {similarity:.4f}\n")
                    
                except KeyError:
                    # 万が一、辞書に登録されていない単語があった場合のエラー回避
                    f_out.write(f"{line} OOV 0.0000\n")

print(f"実験完了！結果を '{output_file}' に保存しました。")