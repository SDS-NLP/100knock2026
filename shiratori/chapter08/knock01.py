from gensim.models import KeyedVectors
import numpy as np


def main():
    model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz", binary=True)

    vocab_size = len(model.key_to_index)
    embedding_dim = model.vector_size

    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

    word_to_id = {"<PAD>": 0}
    id_to_word = {0: "<PAD>"}

    for i, word in enumerate(model.key_to_index, start=1):
        embedding_matrix[i] = model[word]
        word_to_id[word] = i
        id_to_word[i] = word

    print("V =", vocab_size)
    print("d_emb =", embedding_dim)
    print("embedding_matrix.shape =", embedding_matrix.shape)

    print("word_to_id['<PAD>'] =", word_to_id["<PAD>"])
    print("id_to_word[0] =", id_to_word[0])

    first_word = list(model.key_to_index.keys())[0]
    print("最初の単語:", first_word)
    print("そのID:", word_to_id[first_word])
    print("IDから単語:", id_to_word[word_to_id[first_word]])


if __name__ == "__main__":
    main()
