def n_gram(target, n):
    ans = []
    for i in range(len(target) - n + 1):
        ans.append(target[i : i + n])
    return ans

X_str = "paraparaparadise"
Y_str = "paragraph"
X = n_gram(X_str, 2)
Y = n_gram(Y_str, 2)


if __name__ == "__main__":
    print(set(X) | set(Y))
    print(set(X) & set(Y))
    print(set(X) - set(Y))
    if "se" in (set(X) & set(Y)):
        print("se is in X and Y.")
    else:
        print("se is not in X or Y.")