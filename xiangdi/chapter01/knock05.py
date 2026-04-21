def ngram(sequence, n):
    """与えられたシーケンス（文字列やリストなど）から
    n-gramを作る.
    """
    return [sequence[i:i+n] 
            for i in range(len(sequence) - n + 1)]
text = "I am an NLPer"
char_tri = ngram(text, 3)
print(char_tri)
word_double = ngram(text.split(), 2)
print(word_double)