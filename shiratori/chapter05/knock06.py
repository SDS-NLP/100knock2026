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
    初夏をテーマに川柳を10個考えよ
    """
    output = "outputs/chapter05/output06.txt"
    get_reply(prompt, output)
