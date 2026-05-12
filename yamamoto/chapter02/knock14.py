#ファイルの先頭10行に対して、各行の1列目だけを抜き出して表示せよ。確認にはcutコマンドなどを用いよ。

def cut_column(file, n):
    
    with open(file, "r", encoding = "utf-8") as names:
        
        count = 0
        
        for line in names:
            
            list = line.split()
            print(list[n - 1])
            count += 1
            
            if count >= 10:
                
                break 
    
    return        

file = "popular-names.txt"
n = 1

print(cut_column(file, n))  

#head popular-names.txt | cut -f 1    