with open('ans54_full.txt', 'r') as f:
    sem_correct = 0
    sem_total = 0
    syn_correct = 0
    syn_total = 0
    is_semantic = True
    
    for line in f:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(':'):
            if line.startswith(': gram'):
                is_semantic = False
            else:
                is_semantic = True
            continue
            
        words = line.split()
        if len(words) >= 5:
            actual = words[3]
            predicted = words[4]
            is_correct = (actual == predicted)
            
            if is_semantic:
                sem_total += 1
                if is_correct:
                    sem_correct += 1
            else:
                syn_total += 1
                if is_correct:
                    syn_correct += 1

sem_acc = sem_correct / sem_total if sem_total > 0 else 0
syn_acc = syn_correct / syn_total if syn_total > 0 else 0

print(f"意味的アナロジー (Semantic) 正解率: {sem_acc:.3f} ({sem_correct}/{sem_total})")
print(f"文法的アナロジー (Syntactic) 正解率: {syn_acc:.3f} ({syn_correct}/{syn_total})")