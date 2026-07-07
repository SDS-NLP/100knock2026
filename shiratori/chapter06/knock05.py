from gensim.models import KeyedVectors
import numpy as np

# ときなおし

file = "data/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(file, binary=True)

analogy_file = "data/questions-words.txt"


def evaluate_analogy(model, analogy_file):
    semantic_correct = 0
    semantic_total = 0

    syntactic_correct = 0
    syntactic_total = 0

    category = ""

    with open(analogy_file, encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()

            # カテゴリ行
            if line.startswith(":"):
                category = line[2:]
                continue

            a, b, c, d = line.split()
            count += 1
            if count % 10 == 0:
                print(f"{count}件処理済み")

            try:
                pred = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]

                if category.startswith("gram"):
                    syntactic_total += 1
                    if pred == d:
                        syntactic_correct += 1
                else:
                    semantic_total += 1
                    if pred == d:
                        semantic_correct += 1

            except KeyError:
                continue

    return (semantic_correct / semantic_total, syntactic_correct / syntactic_total)


def main():
    semantic_acc, syntactic_acc = evaluate_analogy(model, analogy_file)

    print(f"Semantic Accuracy : {semantic_acc:.4f}")
    print(f"Syntactic Accuracy: {syntactic_acc:.4f}")


if __name__ == "__main__":
    main()

# Semantic Accuracy : 0.7309
# Syntactic Accuracy: 0.7400
