import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using ONLY the context below.
If the answer is not found in the context, say exactly: "I couldn't find that in the document."
Do not make up or infer information beyond what is written.

CONTEXT:
{context}"""


class LLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def generate(
        self, user_message: str, context: str, history: list[dict]
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
