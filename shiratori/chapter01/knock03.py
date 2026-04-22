import re

txt = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

words = txt.split()

result = []

for word in words:
    word = re.sub(r"[^a-zA-Z]", "", word)
    result.append(len(word))

print(result)
