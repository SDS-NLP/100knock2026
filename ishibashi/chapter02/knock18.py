def count_and_sort_data(path):
    """1列目の出現頻度をカウントし、頻度の多い順に並べる関数"""
    data_counts = {}

    with open(path) as f:
        for line in f:
            data = line.split('\t')[0]
            data_counts[data] = data_counts.get(data, 0) + 1
        
    sorted_data = sorted(data_counts.items(), key=lambda x: x[1], reverse=True)
    
    for name, count in sorted_data:
        print(f"{count} {name}")

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    count_and_sort_data(path)

"""
UNIXコマンド
cut -f 1 popular-names.txt | sort | uniq -c | sort -nr
"""