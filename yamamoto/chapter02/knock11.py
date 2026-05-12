#ファイルの先頭N行だけを表示せよ。例えば、N=10として先頭10行を表示せよ。確認にはheadコマンドを用いよ。

def head_N(file, n):
    
    with open(file, "r", encoding = "utf-8") as names:
        
        count = 0
        
        for line in names:
            
            print(line)
            count += 1
            
            if count >= n:
                
                return n                

file = "popular-names.txt"
n = 10

print(head_N(file, n))

#head popular-names.txt