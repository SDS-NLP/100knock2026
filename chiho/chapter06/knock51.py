#“United States”と”U.S.”のコサイン類似度
import os
from pathlib import Path
from gensim.models import KeyedVectors
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float :
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR
data_path = DATA_DIR / "GoogleNews-vectors-negative300.bin"


wordvector = KeyedVectors.load_word2vec_format(data_path, binary=True)

vec1 = wordvector["United_States"]
vec2 = wordvector["U.S."]

print(cosine_sim(vec1, vec2))
