from knock05 import n_gram


def set_operations(x: set, y: set):
    return x | y, x & y, x - y

def main():
    bi = 2
    sentence1 = "paraparaparadise"
    sentence2 = "paragraph"

    X = set(n_gram(sentence1, bi))
    Y = set(n_gram(sentence2, bi))

    union, intersection, difference = set_operations(X, Y)
    print(f"union: {union}\nintersection: {intersection}\ndifference: {difference}")
    print("-"*60)

    target = "se"
    if target in X:
        print(f"{target} は X に含まれています")
    else:
        print(f"{target} は X に含まれていません")
    if target in Y:
        print(f"{target} は Y に含まれています")
    else:
        print(f"{target} は Y に含まれていません")

if __name__ == "__main__":
    main()


