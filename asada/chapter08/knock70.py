import gensim
import numpy as np


class WordEmbeddingToolkit:
    def __init__(self, model_path="./GoogleNews-vectors-negative300.bin"):
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            model_path, binary=True
        )
        V, d_emb = word2vec_model.vectors.shape
        self.matrix = np.zeros((V + 1, d_emb))
        self.matrix[1:] = word2vec_model.vectors
        self.idx_to_token = {0: "<PAD>"}
        self.token_to_idx = {"<PAD>": 0}
        for token, idx in word2vec_model.key_to_index.items():
            self.idx_to_token[idx + 1] = token
            self.token_to_idx[token] = idx + 1

    def convert_tokens_to_ids(self, tokens: list) -> list:
        """
        単語埋め込みの語彙でカバーされていない単語はスキップ
        """
        return [
            self.token_to_idx[token] for token in tokens if token in self.token_to_idx
        ]

    def get_mean_vector(self, ids: list):
        """
        単語埋め込みの平均ベクトルでテキストの特徴ベクトルを表現
        """
        return np.mean(self.matrix[ids], axis=0)


if __name__ == "__main__":
    toolkit = WordEmbeddingToolkit()
    E = toolkit.matrix
    print(f"単語埋め込み行列Eの形状: {E.shape}")
