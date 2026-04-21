def ngram(sequence, n):
    """与えられたシーケンス（文字列やリストなど）から
    n-gramを作る.
    """
    return [sequence[i:i+n] 
            for i in range(len(sequence) - n + 1)]
x = set(ngram("paraparaparadise", 2))
y = set(ngram("paragraph", 2))
print(x)
print(y)
union = x | y
print(union)
intersection = x & y
print(intersection)
difference = x - y
print(difference)
print("'se' in x:", 'se' in x)
print("'se' in y:", 'se' in y)