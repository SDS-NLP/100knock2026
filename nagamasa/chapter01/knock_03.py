str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

str_removed = str.replace(",","").replace(".","")

word_list = str_removed.split(" ")

count_list = [len(word) for word in word_list]

print(count_list)