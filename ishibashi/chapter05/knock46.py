import os
from dotenv import load_dotenv
from groq import Groq

def generate_senryu():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_id = 'llama-3.3-70b-versatile'

    theme = "月"
    prompt = f"""
    「{theme}」をお題にして、ユーモラスで面白い川柳の案を10個作成してください。1から10まで番号を振って出力してください。
    たとえば、「月よりも　君を見ていて　躓いた」などが想定される川柳です。
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "あなたはユーモアのセンスがある優秀な川柳作家です。5・7・5の音のリズムを厳格に守って作成してください。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model_id,
            temperature=0.7, 
        )
        
        response_text = chat_completion.choices[0].message.content
        print(response_text)
        
    except Exception as e:
        print(f"エラー発生: {e}")

if __name__ == "__main__":
    generate_senryu()