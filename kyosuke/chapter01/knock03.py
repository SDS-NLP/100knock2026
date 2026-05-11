text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
clean_text = text.replace(",","").replace(".","")
words = clean_text.split()
print([len(word) for word in words])