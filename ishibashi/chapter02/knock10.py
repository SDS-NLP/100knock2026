def count_lines(path):
    """fileの行数を数える関数"""
    lines_cnt = 0

    with open(path) as f:
        for line in f:
            lines_cnt += 1
        
    return lines_cnt

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'
    
    print(count_lines(path))

"""
UNIXコマンド
wc -l popular-names.txt
"""