from gensim.models import KeyedVectors

def evaluate_accuracies():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'
    input_file = './chapter06/questions-words.txt'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    score, sections = model.evaluate_word_analogies(input_file)

    semantic_correct = 0
    semantic_total = 0
    syntactic_correct = 0
    syntactic_total = 0

    for section in sections:
        section_name = section['section']
        correct_count = len(section['correct'])
        incorrect_count = len(section['incorrect'])
        total_count = correct_count + incorrect_count

        if total_count == 0:
            continue

        if section_name.startswith('gram'):
            syntactic_correct += correct_count
            syntactic_total += total_count
        
        elif section_name != 'Total accuracy':
            semantic_correct += correct_count
            semantic_total += total_count
    
    if semantic_total > 0:
        sem_acc = semantic_correct / semantic_total
        print(f"意味的アナロジー (Semantic) 正解率: {sem_acc:.4f} ({semantic_correct}/{semantic_total})")
        
    if syntactic_total > 0:
        syn_acc = syntactic_correct / syntactic_total
        print(f"文法的アナロジー (Syntactic) 正解率: {syn_acc:.4f} ({syntactic_correct}/{syntactic_total})")

if __name__ == "__main__":
    evaluate_accuracies()