import zipfile
import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import spearmanr

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"
zip_path = "tmp/wordsim353.zip"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

# zip内のcombined.csvを探して読み込む
with zipfile.ZipFile(zip_path) as z:
    file_names = z.namelist()
    print("zip内のファイル:")
    for name in file_names:
        print(name)

    target_file = None
    for name in file_names:
        if name.endswith("combined.csv"):
            target_file = name
            break

    if target_file is None:
        raise FileNotFoundError("combined.csv が zip ないですね。")

    with z.open(target_file) as f:
        df = pd.read_csv(f)

print(df.head())
print(df.columns)