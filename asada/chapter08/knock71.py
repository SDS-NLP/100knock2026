import gensim
import polars as pl
import torch

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)


def splitter(file_path: str):
    return (
        pl.scan_csv(file_path, separator="\t")
        .with_columns(
            pl.col("sentence")
            .str.to_lowercase()
            .str.strip_chars()
            .str.split(" ")
            .alias("words")
        )
        .collect()
    )


def vocabs_constructor(file_path: str):
    vocabs = set()
    df = splitter(file_path)
    for (words,) in df.select("words").iter_rows():
        vocabs.update(words)

    return vocabs


def token2id_constructor(vocabs: set):
    token_to_id = {"<PAD>": 0}
    for token in vocabs:
        if token in model.key_to_index:
            token_to_id[token] = len(token_to_id)
    return token_to_id


def objectify(file_path: str):
    result = []
    df = splitter(file_path)

    for row in df.iter_rows():
        sentence, label, words = row
        ids = [token_to_id[word] for word in words if word in token_to_id]
        result.append(
            {
                "text": sentence,
                "label": torch.tensor(float(label)),
                "input_ids": torch.tensor(ids),
            }
        )
    return result


# それぞれのデータセットの最初の事例で動作確認
if __name__ == "__main__":
    train = "SST-2/train.tsv"
    dev = "SST-2/dev.tsv"
    vocabs = vocabs_constructor(train)
    vocabs.update(vocabs_constructor(dev))
    token_to_id = token2id_constructor(vocabs)
    print(objectify(train)[0])

# result
# {'text': 'hide new secretions from the parental units ', 'label': te
# nsor(0.), 'input_ids': tensor([ 9411,   480,  3389,  4806,  3112,  2
# 532, 12884])}

print(objectify(dev)[0])

# result
# {'text': "it 's a charming and often affecting journey . ", 'label':
#  tensor(1.), 'input_ids': tensor([ 1875, 11709,  6945,  6527,  1147]
# )}
