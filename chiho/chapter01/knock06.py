def n_gram(seq, n):
    return [seq[i:i+n] for i in range(len(seq)-n+1)]

X = set(n_gram("paraparaparadise", 2))
Y = set(n_gram("paragraph", 2))

print( X | Y )
print( X & Y )
print( X - Y )

for name, S in [("X", X), ("Y", Y)]:
    if "se" in S:
        print(f"{name}にはseが含まれます")
    else:
        print(f"{name}にはseは含まれません")
