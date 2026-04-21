def n_gram(sentence, n):
    n_gram_list = []
    
    for i in range(0, len(sentence)-n+1):
        n_gram_list.append(sentence[i:i+n])
    
    return n_gram_list

def pre(sentence):
    treated = sentence.replace(",","").replace(".","").replace(" ","")
    
    return treated


if __name__ == "__main__":
    print(n_gram(pre("I am an NLPer"), 3))

    print(n_gram(["I", "am", "an", "NLPer"],2))