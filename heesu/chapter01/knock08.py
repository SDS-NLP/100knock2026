def f(s):
    return "".join([chr(219-ord(c)) if (ord(c) > 96 and ord(c) < 123) else c for c in s])


print(f("This is a secret message"))
print(f("これは普通のメセージ"))
