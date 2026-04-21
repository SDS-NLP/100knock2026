import knock05

target1 = "paraparaparadise"
target2 = "paragraph"

X = set(knock05.n_gram(target1, 2))
Y = set(knock05.n_gram(target2, 2))

union_set = X | Y
intersection_set = X & Y
difference_set = X - Y

print("X:", X)
print("Y:", Y)

print("union_set:", union_set)
print("intersection_set:", intersection_set)
print("difference_set:", difference_set)

if "se" in X:
    print("True: se in X")
else:
    print("False: se not in X")

if "se" in Y:
    print("True: se in Y")
else:
    print("False: se not in Y")