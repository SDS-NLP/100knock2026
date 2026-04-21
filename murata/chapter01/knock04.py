s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
sl = s.split(" ")
e_d = dict()
for i in range(len(sl)):
    if i+1 in (1, 5, 6, 7, 8, 9, 15, 16, 19):
        e_d[sl[i][0]] = i+1
    else:
        e_d[sl[i][0] + sl[i][1]] = i+1
print(e_d)