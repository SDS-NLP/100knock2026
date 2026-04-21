sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

one = [1, 5, 6, 7, 8, 9, 15, 16, 19]

print(
    {
        (i[0] if j + 1 in one else i[:2]): j + 1
        for j, i in enumerate(sentence.replace(".", "").split())
    }
)
