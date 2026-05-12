text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

text = text.split()

numbers_list = [1,5,6,7,8,9,15,16,19]

answer = {}

id = 1
for i in text:
    if id in numbers_list:
        answer[i[:1]] = id

    else:
        answer[i[:2]] = id
    id += 1

print(answer)