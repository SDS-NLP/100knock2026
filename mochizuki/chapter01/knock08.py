def cipher(text):
    ans = ""
    for c in text:
        if c.islower():
            ans += chr(219 - ord(c))
        else:
            ans += c
    return ans


str1 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
ans1 = cipher(str1)
ans2 = cipher(ans1)

if __name__ == "__main__":
    print(ans1)
    print(ans2)
