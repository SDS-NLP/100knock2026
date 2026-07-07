from pathlib import Path
import argparse

import torch
import joblib
from gensim.models import KeyedVectors


def create_embedding_matrix(limit=500000):
    model_path='./chapter08/GoogleNews-vectors-negative300.bin.gz'

    if not model_path.exists():
        print(f"エラー: 単語ベクトルのファイルが見つかりません: {model_path}")
        return

    w2v = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=limit)
    
    vocab_size = len(w2v.index_to_key)
    embedding_dim = w2v.vector_size
    
    matrix_rows = vocab_size + 1
    print(f"   -> 読み込み完了。語彙数: {vocab_size}, 次元数: {embedding_dim}")
    print(f"   -> 埋め込み行列の最終サイズ: {matrix_rows} x {embedding_dim}")

    embedding_matrix = torch.zeros((matrix_rows, embedding_dim), dtype=torch.float32)

    word2id = {'<PAD>': 0}
    id2word = {0: '<PAD>'}

    for i, word in enumerate(w2v.index_to_key, start=1):
        word2id[word] = i
        id2word[i] = word
        embedding_matrix[i] = torch.from_numpy(w2v[word])


    torch.save(embedding_matrix, './chpater08/embedding_matrix.pt')
    joblib.dump(word2id, './chapter08/word2id.joblib')
    joblib.dump(id2word, './chapter08/id2word.joblib')
    

    print(f"行 0 (<PAD>) のベクトル (最初の5次元): {embedding_matrix[0][:5]}")
    print(f"行 1 ({id2word[1]}) のベクトル (最初の5次元): {embedding_matrix[1][:5]}")
    print(f"word2id['<PAD>']: {word2id['<PAD>']}")
    print(f"id2word[0]:       {id2word[0]}")
    print(f"word2id['the']:   {word2id.get('the', 'Not Found')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='事前学習済み単語埋め込みから埋め込み行列を作成する')
    parser.add_argument('--limit', type=int, default=500000, help='読み込む語彙数')
    args = parser.parse_args()

    create_embedding_matrix(limit=args.limit)