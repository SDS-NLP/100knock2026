import hashlib


def get_sha256(path):
    with open(path, "rb") as fp:
        return hashlib.sha256(fp.read()).hexdigest()


is_valid = True
for filename in [f"xa{chr(ord('a') + a)}" for a in range(10)]:
    if get_sha256(filename) != get_sha256(f"py_{filename}"):
        print(f"failed on {filename}")
        is_valid = False
        break
if is_valid:
    print("Validation complete, same files produced by the python script and split.")
else:
    print("Validation failed.")
