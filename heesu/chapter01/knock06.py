from knock05 import ngram

str0 = "paraparaparadise"
str1 = "paragraph"

bigram0 = ngram(str0, 2)
bigram1 = ngram(str1, 2)

X = set(bigram0)
Y = set(bigram1)

union = X | Y
intersection = X & Y
diff = X - Y

print(f"Union: {union}\nIntersection: {intersection}\nDifference: {diff}")

if 'se' in X:
    print('The bigram "se" is in set X')
else:
    print('The bigram "se" is not in set X')

if 'se' in Y:
    print('The bigram "se" is in set Y')
else:
    print('The bigram "se" is not in set Y')
