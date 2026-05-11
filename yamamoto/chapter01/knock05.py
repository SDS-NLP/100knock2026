def n_gram(sequence, n):
    
    result = []
    
    for i in range(len(sequence) - n + 1):
        
        result.append(sequence[i:(i + n)]) #i番目からn文字を取り出してresultに格納
        
    return result

sequence = "I am an NLPer"

print("bi_gram :", n_gram(sequence, 2))
print("tri_gram :", n_gram(sequence, 3))