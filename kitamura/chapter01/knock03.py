tx = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
ans_list = []

tx = tx.split()

for i in tx:
    ans_list.append(len(i))

print(ans_list)
