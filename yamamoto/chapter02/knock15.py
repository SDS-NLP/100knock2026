#ファイルを行単位でN分割し、別のファイルに格納せよ。例えば、N=10としてファイルを10分割せよ。同様の処理をsplitコマンドで実現せよ。

def split_file(file, n):
    
    with open(file, "r", encoding = "utf-8") as names:
        
        count = 0
        
        while count <= n:
            
            with open(f"file{float(count / n)}.txt", "w", encoding = "utf-8") as F:
            
                for line in names:
                
                    F.write(line)
                    count += 1
                
                    if count % n:
                        
                        break

file = "popular-names.txt"
n = 10

split_file(file, n)           