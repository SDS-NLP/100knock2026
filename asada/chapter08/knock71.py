import polars as pl

from knock70 import WordEmbeddingToolkit


def tokenize(file_path: str, toolkit):
    """
    テキストの全トークンが単語埋め込みの語彙に含まれておらず、空のトークン列となってしまう事例は削除する
    """
    df = (
        pl.scan_csv(file_path, separator="\t")
        .with_columns(
            pl.col("sentence")
            .str.to_lowercase()
            .str.split(by=" ")
            .map_elements(toolkit.convert_tokens_to_ids, return_dtype=pl.List(pl.Int64))
            .alias("input_ids")
        )
        .filter(pl.col("input_ids").list.len() != 0)
        .with_columns(
            pl.col("input_ids")
            .map_elements(toolkit.get_mean_vector, return_dtype=pl.List(pl.Float32))
            .alias("mean_vector")
        )
        .collect()
    )
    return df


def register_wordex(input_ids: list):
    wordex = set()
    for ids in input_ids:
        wordex.update(ids)
    return wordex


def remap_ids(input_ids, id_map):
    return [[id_map[old] for old in ids] for ids in input_ids]


if __name__ == "__main__":
    toolkit = WordEmbeddingToolkit()
    TRAIN_PATH = "SST-2/train.tsv"
    df_train = tokenize(TRAIN_PATH, toolkit)
    print(df_train.head())
    train_input_ids = df_train.get_column("input_ids").to_list()
    train_wordex = register_wordex(train_input_ids)
    print(f"訓練セットで使われる語彙数: {len(train_wordex)}")  # 12346
