import re
from collections import Counter
import MeCab
import matplotlib.pyplot as plt
import numpy as np

def clean_text(text):
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'｜', '', text)

    return text

def plot_zipf_graph(file, output='zipf_law.png'):
    """指定したファイルの全単語からZipfの法則のグラフを描画する関数"""
    tagger = MeCab.Tagger()
    word_counter = Counter()

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        cleaned_text = clean_text(text)

        parsed_lines = tagger.parse(cleaned_text).split('\n')
        for line in parsed_lines:
            if line == 'EOS' or line == '':
                continue

            parts = line.split('\t')
            word = parts[0].strip()
            if word:
                word_counter[word] += 1

    frequencies = [count for word, count in word_counter.most_common()]
    ranks = np.arange(1, len(frequencies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, frequencies, label='Acutual Word Frequency')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Rank (Log Scale)')
    plt.ylabel('Frequency (Log Scale)')
    plt.title("Zipf's Law")

    plt.grid(True, which='both', ls='--')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    file = './chapter04/kokoro.txt'
    plot_zipf_graph(file)