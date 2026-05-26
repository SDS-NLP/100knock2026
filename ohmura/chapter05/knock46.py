from google import genai

client = genai.Client()

prompt = """
エンジニアの日常やあるあるをお題にして、面白い川柳を「5文字 7文字 5文字」の形式で10個作成してください。
日本語の音の響きが綺麗に五・七・五（上五が5文字、中七が7文字、下五が5文字）になるように厳密に作成してください。

1行に1つの川柳を書いて、1から10までの番号を振って出力してください。
余計な解説や挨拶は一切含めないでください。
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
)

print("\n--- 生成されたエンジニア川柳 10選 ---")
print(response.text)