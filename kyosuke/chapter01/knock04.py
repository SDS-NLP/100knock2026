text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
clean_text = text.replace(",","").replace(".","")
words = clean_text.split()

n = [1, 5, 6, 7, 8, 9, 15, 16, 19]
number_list = [2]*len(words)
for i in n:
    number_list[i-1] = 1

ans = {}
for i in range(len(words)):
    ans[words[i][:number_list[i]]] = i+1
print(ans)