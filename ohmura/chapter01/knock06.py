def n_gram(target, n):
    result = []

    for i in range(len(target) - n + 1):
        result.append(target[i:i + n])
    return result

str1 = "paraparaparadise"
str2 = "paragraph"

X = set(n_gram(str1, 2))
Y = set(n_gram(str2, 2))

print(f"X: {X}")
print(f"Y: {Y}")

union = X | Y
intersection = X & Y 
difference = X - Y

print(f"和集合: {union}")
print(f"積集合: {intersection}")
print(f"差集合: {difference}")

print(f"'se' in X: {'se' in X}")
print(f"'se' in Y: {'se' in Y}")