import polars as pl


def label_counter(file_path: str) -> tuple[int, int]:
    return (
        pl.scan_csv(file_path, separator="\t")
        .select(positive=pl.col("label").sum(), negative=(pl.col("label") == 0).sum())
        .collect()
        .row(0)
    )


train_pos, train_neg = label_counter("SST-2/train.tsv")
print(f"学習データ\npositive: {train_pos}件, negative: {train_neg}件")

dev_pos, dev_neg = label_counter("SST-2/dev.tsv")
print(f"検証データ\npositive: {dev_pos}件, negative: {dev_neg}件")
