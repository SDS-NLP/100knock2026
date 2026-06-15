#54の実行結果を用い、意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ。

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary = True
)

semantic_count = 0 #意味的アナロジーの数
semantic_score = 0 #意味的アナロジーの正解数

syntactic_count = 0 #文法的アナロジーの数
syntactic_score = 0 #文法的アナロジーの正解数

with open("questions-words.txt", "r", encoding = "utf-8") as file:
    
    gram_flag = False #文法的アナロジーかどうかの判定
    
    for line in file:
        
        line = line.strip()
        
        if line.startswith(":"):
            
            print(line)
             
            if line.startswith(": gram"): #文法的アナロジーのときTrue
            
                gram_flag = True
            
                continue
            
            else:
                
                gram_flag = False
                
                continue
        
        word1, word2, word3, word4 = line.split()
        
        if word1 not in model or word2 not in model or word3 not in model or word4 not in model:
            
            continue
        
        result = model.most_similar( #この書き方だとresult候補からword1~3は除外される
            positive = [word2, word3],
            negative = [word1],
            topn = 1)
        
        pred_word, similarity = result[0] #resultの要素は(単語, 類似度)
        
        if gram_flag == True:
            
            syntactic_count += 1
            
            if pred_word == word4:
                
                syntactic_score += 1
            
        else:
            
            semantic_count += 1
            
            if pred_word == word4:
                
                semantic_score += 1

semantic_accuracy = semantic_score / semantic_count #意味的アナロジーの正解率
syntactic_accuracy = syntactic_score / syntactic_count #文法的アナロジーの正解率

print(semantic_score, semantic_count)
print(syntactic_score, syntactic_count)

print("semantic analogy:", semantic_accuracy)
print("syntactic analogy:", syntactic_accuracy)