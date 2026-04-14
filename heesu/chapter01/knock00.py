str0 = "パトカー"
str1 = "タクシー"

assert len(str0) == len(str1), "The script doesn't account for mismatch in string length between str0 and str1"

out_str = ""
for i in range(len(str0)):
    out_str.append(str0[i])
    out_str.append(str1[i])

if __name__ == "__main__":
    print(out_str)


