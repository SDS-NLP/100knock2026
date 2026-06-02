import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai
import os
from dotenv import load_dotenv


def solve_historical_problem():
    """
    Gemini APIを使用してzero-shot推論で歴史問題を解く
    """

    load_dotenv()
    
    # APIキーの設定
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません")
    
    genai.configure(api_key=api_key)
    
    # モデルの初期化
    model = genai.GenerativeModel("gemini-2.5-flash-lite") 
    
    # 問題のプロンプト
    prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。

以下の形式で回答してください：
1. 正しい順序（例：ア→イ→ウ）
2. 各出来事の年代
3. 簡潔な説明
"""
    response = model.generate_content(prompt)

    # Zero-shot推論
    print("=" * 60)
    print("問題：9世紀のできごとを年代順に並べよ")
    print("=" * 60)
    print("\n問題文：")
    print("ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。")
    print("イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。")
    print("ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。")
    print("\n" + "=" * 60)
    print("Gemini (Zero-shot推論) による解答：")
    print("=" * 60 + "\n")

    print(response.text)


if __name__ == "__main__":
    solve_historical_problem()
