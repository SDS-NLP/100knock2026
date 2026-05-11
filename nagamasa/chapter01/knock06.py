from nagamasa.chapter01.knock05 import n_gram

X = n_gram("paraparaparadise",2)
Y = n_gram("paragraph",2)

X = set(X)
Y = set(Y)

print(X | Y)
print(X & Y)
print(X - Y)

print("Xに含まれるか")
print("se" in X)

print("Yに含まれるか")
print("se" in Y)