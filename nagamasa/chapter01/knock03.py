text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

text_removed = text.replace(",","").replace(".","")

word_list = text_removed.split(" ")

count_list = [len(word) for word in word_list]

print(count_list)