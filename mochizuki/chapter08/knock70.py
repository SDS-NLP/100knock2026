import gzip
import json
import numpy as np

W2V_PATH = 'GoogleNews-vectors-negative300.bin.gz'
LIMIT = 1000000  # 語彙を削減してメモリを節約

def load_word2vec(path, limit=None):
    with gzip.open(path, 'rb') as f:
        vocab_size, dim = map(int, f.readline().split())
        if limit is not None:
            vocab_size = min(vocab_size, limit)

        # 先頭行 (id=0) は <PAD> 用のゼロベクトルとして予約
        W = np.zeros((vocab_size + 1, dim), dtype=np.float32)
        token2id = {'<PAD>': 0}
        id2token = ['<PAD>']

        binary_len = np.dtype(np.float32).itemsize * dim
        for i in range(vocab_size):
            word = bytearray()
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    word += ch
            vec = np.frombuffer(f.read(binary_len), dtype=np.float32)
            idx = i + 1
            W[idx] = vec
            token2id[word.decode('utf-8', errors='ignore')] = idx
            id2token.append(word.decode('utf-8', errors='ignore'))
        return W, token2id, id2token

def main():
    W, token2id, id2token = load_word2vec(W2V_PATH, LIMIT)
    print(f'embedding matrix shape: {W.shape}')  # (V+1, d)
    print(f'row 0 (<PAD>) is zero vector: {not W[0].any()}')

    np.save('embeddings.npy', W)
    with open('token2id.json', 'w') as f:
        json.dump(token2id, f)
    print('saved: embeddings.npy, token2id.json')

if __name__ == '__main__':
    main()
