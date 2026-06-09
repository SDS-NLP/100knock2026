file_path = 'GoogleNews-vectors-negative300.bin.gz'
word_file = 'knock54_output.txt'
semantic_total = 0
semantic_count = 0
syntactic_total = 0
syntactic_count = 0
is_gram = False

with open(word_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith(":"):
            if line.startswith(": gram"):
                is_gram = True
            else:
                is_gram = False
            continue

        words = line.split()
        if is_gram:
            syntactic_total += 1
            if words[3] == words[4]:
                syntactic_count += 1
        else:
            semantic_total += 1
            if words[3] == words[4]:
                semantic_count += 1


print("Semantic accuracy:"+str(semantic_count/semantic_total))
print("Syntactic accuracy:"+str(syntactic_count/syntactic_total))