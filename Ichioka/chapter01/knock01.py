s1 = "パトカー"
s2 = "タクシー"

interleaved_text = ""
result = ""

for i in range(min(len(s1), len(s2))):
    interleaved_text += s1[i] + s2[i]

print(f"抽出対象{interleaved_text}")

# 偶数を取る

for i in range(1, len(interleaved_text), 2):
    result += interleaved_text[i]

print(f"抽出結果{result}")


