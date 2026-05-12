import re

sentence = (
    "Hi He Lied Because Boron Could Not Oxidize Fluorine. "
    "New Nations Might Also Sign Peace Security Clause. "
    "Arthur King Can."
)
sentence = re.sub(r"[,.]", "", sentence)
word_set = sentence.split()
index_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]
result = {}
for i in range(len(word_set)):
    if i + 1 in index_list:
        key = word_set[i][0]
    else:
        key = word_set[i][:2]
    result[key] = i + 1
print(result)
