sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

print([len(i) for i in sentence.replace(",", "").replace(".", "").split()])
