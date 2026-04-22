txt = """
Hi He Lied Because Boron Could Not Oxidize Fluorine. 
New Nations Might Also Sign Peace Security Clause. Arthur King Can.
"""

words = txt.split()

ini_positions = [1, 5, 6, 7, 8, 9, 15, 16, 19]

result = {}

for i, word in enumerate(words, start=1):
    if i in ini_positions:
        key = word[0]
    else:
        key = word[:2]

    result[key] = i


print(result)
