from gensim.models import KeyedVectors
import numpy as np
import pickle
import pandas as pd

google_news = "data/GoogleNews-vectors-negative300.bin.gz"
train_path = "data/SST-2/train.tsv"
dev_path = "data/SST-2/dev.tsv"


def main():
    train = pd.read_csv(train_path, sep="\t")
    dev = pd.read_csv(dev_path, sep="\t")

    # SST-2に出現する単語を収集
    vocab = set()

    for text in train["sentence"]:
        vocab.update(text.split())

    for text in dev["sentence"]:
        vocab.update(text.split())

    print(f"SST-2語彙数: {len(vocab)}")

    model = KeyedVectors.load_word2vec_format(
        google_news,
        binary=True,
    )

    # <PAD>を追加
    word_to_id = {"<PAD>": 0}
    id_to_word = {0: "<PAD>"}

    embedding_list = [np.zeros(model.vector_size, dtype=np.float32)]

    # SST-2にある単語だけ追加
    for word in sorted(vocab):
        if word in model:
            word_to_id[word] = len(word_to_id)
            id_to_word[len(id_to_word)] = word
            embedding_list.append(model[word])

    # 埋め込み行列作成
    embedding_matrix = np.array(embedding_list, dtype=np.float32)

    V = len(word_to_id)
    d_emb = embedding_matrix.shape[1]

    # 確認
    print("V =", V)
    print("d_emb =", d_emb)
    print("embedding_matrix.shape =", embedding_matrix.shape)

    print("word_to_id['<PAD>'] =", word_to_id["<PAD>"])
    print("id_to_word[0] =", id_to_word[0])

    # <PAD>以外の最初の単語を確認
    first_word = list(word_to_id.keys())[1]
    print("最初の単語:", first_word)
    print("そのID:", word_to_id[first_word])
    print("IDから単語:", id_to_word[word_to_id[first_word]])

    np.save("data/embedding_matrix.npy", embedding_matrix)

    with open("data/sst2_word_to_id.pkl", "wb") as f:
        pickle.dump(word_to_id, f)

    with open("data/sst2_id_to_word.pkl", "wb") as f:
        pickle.dump(id_to_word, f)


if __name__ == "__main__":
    main()
