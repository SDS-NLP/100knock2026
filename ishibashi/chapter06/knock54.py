from gensim.models import KeyedVectors

def run_analogy_experiment():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'
    input_file = './chapter06/questions-words.txt'
    output_file = './chapter06/knock54_output.txt'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    count = 0
    is_target_section = False

    with open(input_file, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()

            if line.startswith(':'):
                if line == ': capital-common-countries':
                    is_target_section = True
                else:
                    is_target_section = False
                continue

            if not is_target_section:
                continue

            words = line.split()
            if len(words) != 4:
                continue

            word1, word2, word3, word4 = words

            try:
                ans = model.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0]

                predicted_word = ans[0]
                similarity = ans[1]

                f_out.write(f"{line} {predicted_word} {similarity:.4f}\n")
                count += 1

            except KeyError as e:
                pass

if __name__ == "__main__":
    run_analogy_experiment()