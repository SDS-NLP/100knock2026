import os
from google import genai


def get_reply(prompt1, prompt2, output):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chat = client.chats.create(model="gemini-2.5-flash-lite")

    with open(output, "w") as f:
        f.write("results\n\n")

    response1 = chat.send_message(prompt1)
    print(response1.text)
    response2 = chat.send_message(prompt2)
    print(response2.text)

    with open(output, "a") as f:
        f.write(response1.text)
        f.write(response2.text)


if __name__ == "__main__":
    prompt1 = """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
    """
    prompt2 = """さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
    """
    output = "outputschapter05/output05.txt"
    get_reply(prompt1, prompt2, output)
