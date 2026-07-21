"""70. 単語埋め込みの読み込み

chapter06 にある Google News の学習済み単語ベクトルを読み込み、
<PAD> を 0 番に予約した埋め込み行列と、
トークン ID との双方向マッピングを構築する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors


SCRIPT_DIR = Path(__file__).resolve().parent
VECTOR_PATH = SCRIPT_DIR.parent / "chapter06" / "GoogleNews-vectors-negative300.bin"
PAD_TOKEN = "<PAD>"


def load_embedding_matrix(
    vector_path: Path = VECTOR_PATH,
) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    """Load pretrained embeddings and reserve ID 0 for padding."""
    word_vectors = KeyedVectors.load_word2vec_format(vector_path, binary=True)
    vocab_size = len(word_vectors.key_to_index) + 1
    emb_dim = word_vectors.vector_size

    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    token_to_id: dict[str, int] = {PAD_TOKEN: 0}
    id_to_token: list[str] = [PAD_TOKEN]

    for token, original_index in word_vectors.key_to_index.items():
        token_id = original_index + 1
        embedding_matrix[token_id] = word_vectors[token]
        token_to_id[token] = token_id
        id_to_token.append(token)

    return torch.from_numpy(embedding_matrix), token_to_id, id_to_token


def main() -> None:
    embedding_matrix, token_to_id, id_to_token = load_embedding_matrix()

    print(f"vector path: {VECTOR_PATH}")
    print(f"embedding_matrix.shape: {tuple(embedding_matrix.shape)}")
    print(f"PAD token id: {token_to_id[PAD_TOKEN]}")
    print(f"id_to_token[0]: {id_to_token[0]}")
    if "United_States" in token_to_id:
        token_id = token_to_id["United_States"]
        print(f"token_to_id['United_States']: {token_id}")
        print(f"id_to_token[{token_id}]: {id_to_token[token_id]}")


if __name__ == "__main__":
    main()
