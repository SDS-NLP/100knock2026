s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
sl = s.split(" ")
ans_l = list()
for i in range(len(sl)):
    ans_l.append(len(sl[i]))
print(ans_l)