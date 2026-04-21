import re

sentence = (
    "Now I need a drink, alcoholic of course, after "
    "the heavy lectures involving quantum mechanics."
)
sentence = re.sub(r"[,.]", "", sentence)
word_set = sentence.split()
result = []
for word in word_set:
    result.append(len(word))
print(result)