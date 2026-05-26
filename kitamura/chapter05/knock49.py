from google import genai
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

file = "i_am_neko.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()


response = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents=text
)

print(response.total_tokens)

# 270