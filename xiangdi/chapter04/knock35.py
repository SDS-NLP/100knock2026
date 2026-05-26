import subprocess

text = "メロスは激怒した。"

result = subprocess.run(
    ["cabocha"],
    input=text,
    text=True,
    stdout=subprocess.PIPE
)

print(result.stdout)
