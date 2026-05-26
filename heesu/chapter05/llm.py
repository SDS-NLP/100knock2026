from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in .env")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def chat(
    prompt: str,
    model: str = "openai/gpt-oss-120b:nitro",
    system: str = "You are a helpful assistant.",
    temperature: float = 0.7,
) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def chat_messages(
    messages: list[dict],
    model: str = "openai/gpt-oss-120b:nitro",
    temperature: float = 0.7,
) -> str:
    """Call with a raw messages list for multi-turn conversations."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
