from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt: str):
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    answer = ask_gemini(
        "Explain what Retrieval-Augmenyted Generation is in one paragraph."
    )

    print(answer)