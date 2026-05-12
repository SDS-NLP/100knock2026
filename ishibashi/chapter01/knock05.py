def n_gram(target, n):
    """n-gramで取り出す関数"""
    res = []
    
    for i in range(len(target) - n + 1):
        res.append(target[i:i+n])
    
    return res

if __name__ == "__main__":
    import knock03

    target = "I am an NLPer"
    words = knock03.split_words(target)

    print(n_gram(target, 3))
    print(n_gram(words, 2))