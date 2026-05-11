str1 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
str1 = str1.replace(".", "").replace(",", "")
str1_list = str1.split()
ans = [len(w) for w in str1_list]

if __name__ == "__main__":
    print(ans)