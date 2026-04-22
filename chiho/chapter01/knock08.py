def cipher(x):
    result = ""
    for c in x:
        if c.islower():
            result += chr(219 - ord(c))
        else:
            result += c
    return result

s = "I am a student in Japan!"
enc = cipher(s)
dec = cipher(enc)

print(enc)
print(dec)
