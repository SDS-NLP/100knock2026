#ファイルの末尾N行だけを表示せよ。例えば、N=10として末尾10行を表示せよ。確認にはtailコマンドを用いよ。

def tail(file, n):
    
    with open(file, "r", encoding = "utf-8") as names:
        
        count = 0
        
        for line in range(-1, -n):
            
            print(names[line])
            count += 1
            
            if count >= n:
                
                break
    
    return        
        
file = "popular-names.txt"
n = 10

print(tail(file, n))     

#tail popular-names.txt