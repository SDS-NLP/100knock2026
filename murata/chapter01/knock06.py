from knock05 import n_gram

X = n_gram(2, "paraparaparadise")
Y = n_gram(2, "paragraph")

X_set = set(X)
Y_set = set(Y)
print("和集合: ", X_set | Y_set )
print("積集合: ", X_set & Y_set)
print("差集合: ", X_set - Y_set)