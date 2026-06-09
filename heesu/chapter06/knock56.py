from gensim.models import KeyedVectors
import gensim.downloader as api_info
import os

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

wordsim_path = os.path.join(
    os.path.dirname(__file__),
    '.venv/lib/python3.13/site-packages/gensim/test/test_data/wordsim353.tsv'
)

pearson, spearman, oov_ratio = model.evaluate_word_pairs(wordsim_path)

print(f"Pearson  r : {pearson.statistic:.4f}  (p={pearson.pvalue:.4e})")
print(f"Spearman r : {spearman.statistic:.4f}  (p={spearman.pvalue:.4e})")
print(f"OOV ratio  : {oov_ratio:.4f}")
