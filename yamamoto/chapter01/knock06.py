def n_gram(sequence, n):
    
    result = []
    
    for i in range(len(sequence) - n + 1):
        
        result.append(sequence[i:(i + n)])
        
    return result

sequence1 = "paraparaparadise"
sequence2 = "paragraph"

X = set(n_gram(sequence1, 2)) #list型をset型に変換
Y = set(n_gram(sequence2, 2))

print("X =", X)
print("Y =", Y)

union = X.union(Y) #和集合
intersection = X.intersection(Y) #積集合
difference = X.difference(Y) #差集合

print("union =", union)
print("intersection =", intersection)
print("difference =", difference)

if "se" in X:
    print("X : true")
else:
    print("X : false")

if "se" in Y:
    print("Y : true")
else:
    print("Y : false")            