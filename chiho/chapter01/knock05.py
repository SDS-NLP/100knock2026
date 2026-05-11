#05. n-gram
#与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ。この関数を用い、”I am an NLPer”という文から文字tri-gram、単語bi-gramを得よ。

def n_gram(sent, n):
    return [sent[i:i+n] for i in range(len(sent) - n + 1)]

s = "I am an NLPer"
char_tri = n_gram(s, 3)
word_bi = n_gram(s.split(), 2)

print(char_tri)
print(word_bi)