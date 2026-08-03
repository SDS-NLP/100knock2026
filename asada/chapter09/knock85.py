import polars as pl
from transformers import AutoTokenizer


def tokenize(file_path: str, tokenizer):
    df = pl.read_csv(file_path, separator="\t")
    sentences = df.get_column("sentence").to_list()
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    return df.with_columns(pl.Series("input_ids", tokens["input_ids"]))


if __name__ == "__main__":
    TRAIN_PATH = "SST-2/train.tsv"
    DEV_PATH = "SST-2/dev.tsv"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    df_train = tokenize(TRAIN_PATH, tokenizer)
    df_dev = tokenize(DEV_PATH, tokenizer)
    print(df_train.head())
    print(df_dev.head())

# shape: (5, 3)
# ┌─────────────────────────────────┬───────┬──────────────────┐
# │ sentence                        ┆ label ┆ input_ids        │
# │ ---                             ┆ ---   ┆ ---              │
# │ str                             ┆ i64   ┆ array[i64, 72]   │
# ╞═════════════════════════════════╪═══════╪══════════════════╡
# │ hide new secretions from the p… ┆ 0     ┆ [101, 4750, … 0] │
# │ contains no wit , only labored… ┆ 0     ┆ [101, 2515, … 0] │
# │ that loves its characters and … ┆ 1     ┆ [101, 1115, … 0] │
# │ remains utterly satisfied to r… ┆ 0     ┆ [101, 2606, … 0] │
# │ on the worst revenge-of-the-ne… ┆ 0     ┆ [101, 1113, … 0] │
# └─────────────────────────────────┴───────┴──────────────────┘
# shape: (5, 3)
# ┌─────────────────────────────────┬───────┬──────────────────┐
# │ sentence                        ┆ label ┆ input_ids        │
# │ ---                             ┆ ---   ┆ ---              │
# │ str                             ┆ i64   ┆ array[i64, 61]   │
# ╞═════════════════════════════════╪═══════╪══════════════════╡
# │ it 's a charming and often aff… ┆ 1     ┆ [101, 1122, … 0] │
# │ unflinchingly bleak and desper… ┆ 0     ┆ [101, 8362, … 0] │
# │ allows us to hope that nolan i… ┆ 1     ┆ [101, 3643, … 0] │
# │ the acting , costumes , music … ┆ 1     ┆ [101, 1103, … 0] │
# │ it 's slow -- very , very slow… ┆ 0     ┆ [101, 1122, … 0] │
# └─────────────────────────────────┴───────┴──────────────────┘
