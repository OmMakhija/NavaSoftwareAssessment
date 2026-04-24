import os
from groq import Groq

class CriticAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def run(self, query: str, facts: str, summary: str) -> str:
        prompt = f"""You are an expert Critic and Editor. Your job is to review a drafted answer to ensure it is accurate, complete, and fully grounded in the provided facts.
If the draft is excellent, output it with minimal polish.
If the draft is missing information from the facts or contains unsupported claims, rewrite it completely to fix the flaws.
Return ONLY the final, improved answer. Do not include your review notes.

Query: {query}
Researcher Facts: {facts}
Draft Answer: {summary}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Critic Error: {str(e)}"
