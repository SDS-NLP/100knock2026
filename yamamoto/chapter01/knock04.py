text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

element = text.split()

print(element)

symbol = {} #空の辞書を作成        

for i in range(len(element)):
    
    if i == 0 or i == 4 or i == 5 or i == 6 or i == 7 or i == 8 or i == 14 or i == 15 or i == 18: #元素記号がアルファベット1文字のとき
        
        symbol[element[i][0]] = i + 1   
    else: #アルファベット2文字のとき
        
        symbol[element[i][:2]] = i + 1  
        
print(symbol)              