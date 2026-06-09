import gensim
import polars as pl

df = pl.read_csv("combined.csv")

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)


def calc_similarity(row: dict) -> float:
    w1 = row["Word 1"]
    w2 = row["Word 2"]
    return model.similarity(w1, w2)


df_with_score = df.with_columns(
    pl.struct(["Word 1", "Word 2"])
    .map_elements(calc_similarity, return_dtype=pl.Float64)
    .alias("score")
)

df_ranked = df_with_score.with_columns(
    [
        pl.col("Human (mean)").rank(descending=True).alias("human_rank"),
        pl.col("score").rank(descending=True).alias("score_rank"),
    ]
)

human = df_ranked.get_column("human_rank")
score = df_ranked.get_column("score_rank")
N = len(df_ranked)
spearman = 1 - (6 * sum((human - score) ** 2) / (N * (N**2 - 1)))
print(spearman)
