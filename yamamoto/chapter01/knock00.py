p = "パトカー"
t = "タクシー"

pt = "" #空の文字列

for i in range(max(len(p), len(t))): #文字数が多いほうに合わせる
    
    if i < len(p): #p[i]が存在するとき
        
        pt += p[i]
    
    if i < len(t): #t[i]が存在するとき
        
        pt += t[i]
    
print(pt)    