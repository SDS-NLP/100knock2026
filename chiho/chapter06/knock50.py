# 50. 単語ベクトルの読み込みと表示
#Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル（300万単語・フレーズ、300次元）を
# ダウンロードし、”United States”の単語ベクトルを表示せよ。ただし、”United States”は内部的には
# ”United_States”と表現されていることに注意せよ。

import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
FILE_CANDIDATES = (
    "GoogleNews-vectors-negative300.bin.gz",
    "GoogleNews-vectors-negative300.bin",
)

data_path = next((DATA_DIR / name for name in FILE_CANDIDATES if (DATA_DIR / name).exists()), None)
if data_path is None:
    raise SystemExit(
        f"単語ベクトルファイルが見つかりません: {DATA_DIR / FILE_CANDIDATES[0]} "
        f"または {DATA_DIR / FILE_CANDIDATES[1]}"
    )


def load_vector(path: Path, target_word: str) -> np.ndarray:
    with path.open("rb") as stream:
        _, vector_size = map(int, stream.readline().split())

        while True:
            word_bytes = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise KeyError(f"{target_word} が見つかりませんでした")
                if char == b" ":
                    break
                if char != b"\n":
                    word_bytes.extend(char)

            word = word_bytes.decode("utf-8")
            vector = np.frombuffer(stream.read(4 * vector_size), dtype=np.float32)

            if word == target_word:
                return vector


print(load_vector(data_path, "United_States"))
