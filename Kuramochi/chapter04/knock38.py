from knock36 import get_jawiki_contents, INPUT_PATH
import math
from collections import Counter
from janome.tokenizer import Tokenizer


def extract_tfidf_keywords(nouns, top_n=10):
    tokenizer  = Tokenizer()
    docs_words = []
    
    # 1. すべてのテキストを形態素解析して、単語（基本形）のリストに変換(ただし記号は除く)
    for doc in nouns:
        words = []
        for token in tokenizer.tokenize(doc):
            if token.part_of_speech.split(',')[0] != '記号':
                words.append(token.base_form)
        docs_words.append(words)
        
    total_docs = len(docs_words)
    idf_scores = {}
    
    # 2. 全記事のデータから、各単語のレア度（IDF）を計算
    all_unique_words = set(word for words in docs_words for word in words)
    for word in all_unique_words:
        # その単語が含まれているドキュメント（記事）数をカウント
        doc_count = sum(1 for words in docs_words if word in set(words))
        # IDFの計算
        idf_scores[word] = math.log(total_docs / doc_count) + 1.0
        
    # 3. 各記事ごとに TF-IDF スコアを計算して上位を抽出
    results = []
    for words in docs_words:
        tf_counter = Counter(words)
        total_words = len(words)
        
        tfidf_doc = {}
        for word, count in tf_counter.items():
            tf = count / total_words               # その記事内での出現割合（TF）
            tfidf_doc[word] = tf * idf_scores[word] # TF × IDF
            
        # スコアが高い順に並び替え
        sorted_tfidf = sorted(tfidf_doc.items(), key=lambda x: x[1], reverse=True)
        # 上位 top_n 件だけを結果リストに追加
        results.append(sorted_tfidf[:top_n])
        
    return results


# --- 関数の使い方（テスト実行） ---
if __name__ == '__main__':
    # リスト形式で複数のテキストを関数に渡します
    sample_texts  = get_jawiki_contents(INPUT_PATH)
    tfidf_results = extract_tfidf_keywords(sample_texts, top_n=3)
    
    # 結果の表示
    for i, keywords in enumerate(tfidf_results):
        print(f"【記事 {i+1} のキーワード】")
        for word, score in keywords:
            print(f"  {word}: {score:.4f}")