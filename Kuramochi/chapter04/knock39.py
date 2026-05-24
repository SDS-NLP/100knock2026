import matplotlib.pyplot as plt
from collections import Counter
from janome.tokenizer import Tokenizer
from knock36 import get_jawiki_contents, INPUT_PATH

def plot_word_frequency_loglog(text):
    """
    テキスト内の単語の出現頻度順位と頻度を両対数グラフでプロットする関数
    """
    # 1. 形態素解析と単語の抽出（記号を除外して基本形を取得）
    tokenizer = Tokenizer()
    words = []
    for token in tokenizer.tokenize(text):
        if token.part_of_speech.split(',')[0] != '記号':
            words.append(token.base_form)

    # 2. 単語の出現頻度をカウントし、多い順に並べる
    # most_common() は [(単語A, 100回), (単語B, 80回)...] のようなリストを返します
    word_counts = Counter(words).most_common()

    # 3. グラフ用のX軸（順位）とY軸（頻度）のデータを作成
    ranks = range(1, len(word_counts) + 1)
    frequencies = [count for word, count in word_counts]

    # 4. 両対数グラフのプロット
    plt.figure(figsize=(8, 6))
    
    # plt.plot()の代わりに plt.loglog() を使うことで自動的に両対数スケールになります
    plt.loglog(ranks, frequencies, marker='.', linestyle='None')
    
    plt.title('単語の出現頻度と順位（ジップの法則）')
    plt.xlabel('出現頻度順位（対数スケール）')
    plt.ylabel('出現頻度（対数スケール）')
    
    # 対数グラフが見やすくなるように細かいグリッド線を引く
    plt.grid(True, which="both", ls="--") 
    
    # グラフを画面に表示
    plt.show()


# --- 実行例 ---
if __name__ == '__main__':
    sample_text = get_jawiki_contents(INPUT_PATH)
    plot_word_frequency_loglog(sample_text)