def n_gram(sequence, n):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
X = set(n_gram("paraparaparadise", 2))
Y = set(n_gram("paragraph", 2))
print(f"和集合；{X | Y}")
print(f"積集合；{X & Y}")
print(f"差集合；{X - Y}")

print ("se" in X)
print ("se" in Y)