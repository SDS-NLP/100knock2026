input_file = 'knock54_out.txt'

# 正解数と全体数をカウントするための変数
semantic_correct  = 0
semantic_total    = 0
syntactic_correct = 0
syntactic_total   = 0

current_category = ''

print("正解率を計算しています...")

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        
        # カテゴリの切り替わり行（例: ': gram1-adjective-to-adverb'）
        if line.startswith(':'):
            current_category = line
            continue
            
        words = line.split()
        
        # データ行には 元の4単語 + 予測単語 + スコア の計6要素があるはず
        if len(words) == 6:
            true_word = words[3]
            predicted_word = words[4]
            
            # 正解かどうかを判定（完全一致）
            is_correct = (true_word == predicted_word)
            
            # カテゴリ名が ': gram' で始まるものは「文法的アナロジー」
            if current_category.startswith(': gram'):
                syntactic_total += 1
                if is_correct:
                    syntactic_correct += 1
            # それ以外のカテゴリは「意味的アナロジー」
            else:
                semantic_total += 1
                if is_correct:
                    semantic_correct += 1

# ゼロ割り算エラーを防ぎつつ正解率を計算
semantic_acc = semantic_correct / semantic_total if semantic_total > 0 else 0
syntactic_acc = syntactic_correct / syntactic_total if syntactic_total > 0 else 0

print(f"--- 評価結果 ---")
print(f"意味的アナロジー 正解率: {semantic_acc:.3f} ({semantic_correct} / {semantic_total}問)")
print(f"文法的アナロジー 正解率: {syntactic_acc:.3f} ({syntactic_correct} / {syntactic_total}問)")