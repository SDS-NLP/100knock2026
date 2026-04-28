import re
tx = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
ans_list = []

tx = tx.split()

for word in tx:
    word = re.sub(r"[^a-zA-Z]", "", word)##カンマを取り除く
    ans_list.append(len(word))

print(ans_list)
