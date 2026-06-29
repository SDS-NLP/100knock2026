"""
平均単語埋め込みを特徴量とするロジスティック回帰モデルの定義.
"""

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


def build_embedding_matrix(model_path: str):
    """
    事前学習済み単語ベクトルから埋め込み行列と語彙辞書を構築する.

    Parameters
    ----------
    model_path : str
        Word2Vec バイナリ形式のファイルパス.

    Returns
    -------
    embedding_matrix : np.ndarray
        形状 (V+1, D). 行0はPAD用ゼロベクトル.
    word2id : dict[str, int]
    id2word : dict[int, str]
    """
    print("単語埋め込みモデルを読み込み中...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("読み込み完了.")

    V, D = len(model.index_to_key), model.vector_size
    embedding_matrix = np.zeros((V + 1, D), dtype=np.float32)
    embedding_matrix[1:] = model.vectors

    word2id: dict[str, int] = {"<PAD>": 0}
    id2word: dict[int, str] = {0: "<PAD>"}
    for token_id, word in enumerate(model.index_to_key, start=1):
        word2id[word] = token_id
        id2word[token_id] = word

    return embedding_matrix, word2id, id2word


class AvgEmbeddingClassifier(nn.Module):
    """
    単語埋め込みの平均ベクトルを特徴量とするロジスティック回帰モデル.

    各トークンの埋め込みを平均して D 次元の特徴ベクトルを作り,
    重みベクトルとの内積 + バイアスにシグモイドを適用してポジティブ確率を返す.

    Parameters
    ----------
    embedding_matrix : np.ndarray
        形状 (V+1, D) の埋め込み行列.
    freeze : bool
        True のとき埋め込み行列を固定する.
    """

    def __init__(self, embedding_matrix: np.ndarray, freeze: bool = True):
        super().__init__()
        _, D = embedding_matrix.shape

        # 事前学習済み埋め込み層（freeze=True で勾配計算を無効化）
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze,
            padding_idx=0,
        )
        # 重みベクトル w とバイアス b（D → 1 の線形変換）
        self.fc = nn.Linear(D, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : Tensor (seq_len,)
            トークンID列.

        Returns
        -------
        Tensor (1,)
            ポジティブである確率.
        """
        emb = self.embedding(input_ids)  # (seq_len, D)
        h   = emb.mean(dim=0)            # (D,)  テキストの平均埋め込みベクトル
        out = torch.sigmoid(self.fc(h))  # (1,)
        return out


if __name__ == "__main__":
    model_path = "GoogleNews-vectors-negative300.bin.gz"
    embedding_matrix, word2id, _ = build_embedding_matrix(model_path)

    model = AvgEmbeddingClassifier(embedding_matrix, freeze=True)
    print(model)

    # 埋め込み層は固定なので fc の D+1 パラメータのみ学習対象
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"学習対象パラメータ数: {trainable:,}")

    # 動作確認
    ids = torch.tensor([word2id["good"], word2id["movie"]], dtype=torch.long)
    print(f"サンプル出力 (good, movie): {model(ids).item():.4f}")