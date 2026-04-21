def n_gram(target, n):
    ans = []
    for i in range(len(target) - n + 1):
        ans.append(target[i : i + n])
    return ans


str1 = "I am an NLPer"
str1_list = str1.split()
ans1 = n_gram(str1.replace(" ", ""), 3)
ans2 = n_gram(str1_list, 2)

if __name__ == "__main__":
    print(ans1)
    print(ans2)
