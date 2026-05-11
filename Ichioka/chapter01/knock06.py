def bigrams(s):
    return {s[i:i+2] for i in range(len(s) - 1)}

str1 = "paraparaparadise"
str2 = "paragraph"

X = bigrams(str1)
Y = bigrams(str2)

union = X | Y
intersection = X & Y
difference = X - Y

print("X:", X)
print("Y:", Y)
print("和集合:", union)
print("積集合:", intersection)
print("差集合:", difference)