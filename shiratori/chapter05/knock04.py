import os
from google import genai


def get_reply(prompt, output):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    with open(output, "w") as f:
        f.write("results\n\n")

    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)

    print(response.text)

    with open(output, "a") as f:
        f.write(response.text)


if __name__ == "__main__":
    prompt = """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
    """
    output = "chapter05/output04.txt"
    get_reply(prompt, output)
