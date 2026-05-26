from google import genai

client = genai.Client()

theme = "リモートワーク"
prompt = f"お題「{theme}」で川柳(5・7・5)を10個作ってください。各案は1行ずつ、番号付きで列挙してください。"

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt,
)

print(response.text)
