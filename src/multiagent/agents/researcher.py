import os
from groq import Groq

class ResearcherAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def run(self, query: str, context: str) -> str:
        prompt = f"""You are an expert Researcher. Your ONLY job is to extract factual information from the provided context that is directly relevant to the user's query.
Do not attempt to answer the question or write a final summary. Do not add outside knowledge. 
Return a bulleted list of facts and relevant excerpts.

Query: {query}
Context: {context}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Researcher Error: {str(e)}"
