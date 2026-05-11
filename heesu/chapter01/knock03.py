string = "Now I need a drink, alcoholic of course, \
after the heavy lectures involving quantum mechanics."
string = string.replace(',', '')

words = string.split(' ')
count_list = [len(word) for word in words]
print(count_list)
