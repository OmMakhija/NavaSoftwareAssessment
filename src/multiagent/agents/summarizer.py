import os
from groq import Groq

class SummarizerAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def run(self, query: str, facts: str) -> str:
        prompt = f"""You are an expert Summarizer. Your ONLY job is to write a clear, concise, and structured answer to the user's query using ONLY the facts provided by the Researcher.
Do not invent information. Structure the answer logically.

Query: {query}
Researcher Facts: {facts}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summarizer Error: {str(e)}"
