pt = "パタトクカシーー"

p = ""
t = ""

for i in range(len(pt)):
    
    if i % 2 == 0:
        
        p += pt[i]
        
    else:
        
        t += pt[i]    

print(p)
print(t)        