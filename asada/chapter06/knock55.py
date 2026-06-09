import polars as pl

df = pl.read_csv("result.csv")

# capital-common-countriesセクションは
# 文法的アナロジーではない。
accuracy = (df["word4"] == df["pred_word"]).sum() / len(df)
print(f"意味的アナロジーの正解率: {accuracy:.4f}")
