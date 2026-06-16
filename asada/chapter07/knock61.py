import polars as pl


def splitter(file_path: str):
    df = (
        pl.scan_csv(file_path, separator="\t")
        .with_columns(
            pl.col("sentence").str.strip_chars().str.split(" ").alias("words")
        )
        .collect()
    )

    return df


def objectify(file_path: str):
    result = []
    for row in splitter(file_path).iter_rows():
        sentence, label, words = row
        vec = {}
        for word in words:
            if word not in vec:
                vec[word] = 1
            else:
                vec[word] += 1
        result.append({"text": sentence, "label": label, "feature": vec})
    return result


if "__name__" == "__main__":
    objectify("SST-2/train.tsv")
    objectify("SST-2/dev.tsv")
