str1 = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
str1 = str1.replace(",", "").replace(".", "")
str_list = str1.split()
list = [0, 4, 5, 6, 7, 8, 14, 15, 18]
ans = {}
for i in range(len(str_list)):
    if i in list:
        key = str_list[i][0]
    else :
        key = str_list[i][0] + str_list[i][1]
    ans[key] = i + 1
    
if __name__ == "__main__":
    print(ans)

