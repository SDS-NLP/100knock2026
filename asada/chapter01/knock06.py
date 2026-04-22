from knock05 import generate_ngram

str1, str2 = "paraparaparadise", "paragraph"

X, Y = set(generate_ngram(str1, 2, True)), set(generate_ngram(str2, 2, True))

print(f"和集合: {X | Y}, 積集合: {X & Y}, 差集合(X\Y): {X - Y}")
print(f"'se' in bi-gram: {'se' in (X | Y)}")
