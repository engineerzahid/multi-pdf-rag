from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import os

def test_groq():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say: RAG project Day 1 ready!"}]
    )
    print("Groq API test response:")
    print(response.choices[0].message.content)
    print("\nDay 1 setup complete!")

if __name__ == "__main__":
    test_groq()
