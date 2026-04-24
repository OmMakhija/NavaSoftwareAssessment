import re
import os
import json
from groq import Groq

class Guardrails:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
        self.patterns = {
            "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            "phone": re.compile(r"(\+91[\-\s]?)?[6-9]\d{9}|\b\d{10}\b|\b\d{3}[\-.\s]\d{3}[\-.\s]\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
            "aadhaar": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        }
        
        print("Guardrails initialized")

    def check_pii(self, text: str) -> dict:
        types_found = []
        for name, pattern in self.patterns.items():
            if pattern.search(text):
                types_found.append(name)
        
        return {
            "pii_found": len(types_found) > 0,
            "types": types_found
        }

    def check_off_topic(self, query: str, answer: str, context: str) -> dict:
        prompt = f"""You are a guardrail module. Your job is to determine if the given answer is grounded in the provided context for the user query.
If the answer relies on information outside the context, it is off-topic.

Query: {query}
Context: {context[:500]}
Answer: {answer}

Respond ONLY with valid JSON, no markdown, no backticks, using exactly this schema:
{{"off_topic": true/false, "reason": "short reason"}}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80
            )
            raw_text = response.choices[0].message.content.strip()
            
            # Strip markdown fences if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            data = json.loads(raw_text.strip())
            return {
                "off_topic": bool(data.get("off_topic", False)),
                "reason": str(data.get("reason", ""))
            }
        except Exception:
            return {
                "off_topic": False,
                "reason": "check failed"
            }

    def run(self, query: str, answer: str, context: str) -> dict:
        pii_result = self.check_pii(query + " " + answer)

        # Short-circuit: If PII is found, skip the LLM call entirely
        if pii_result["pii_found"]:
            return {
                "verdict": "PII_DETECTED",
                "pii": pii_result,
                "off_topic": {"off_topic": False, "reason": "skipped"}
            }

        off_topic_result = self.check_off_topic(query, answer, context)

        verdict = "OFF_TOPIC" if off_topic_result["off_topic"] else "SAFE"

        return {
            "verdict": verdict,
            "pii": pii_result,
            "off_topic": off_topic_result
        }

    def format_report(self, result: dict) -> str:
        verdict = result["verdict"]
        
        if verdict == "SAFE":
            verdict_str = "SAFE ✅"
        elif verdict == "OFF_TOPIC":
            verdict_str = "OFF_TOPIC ⚠️"
        elif verdict == "PII_DETECTED":
            verdict_str = "PII_DETECTED 🚨"
        else:
            verdict_str = "BOTH 🚨"
            
        pii_str = "No PII detected."
        if result["pii"]["pii_found"]:
            pii_str = f"PII types found: {', '.join(result['pii']['types'])}"
            
        off_topic_str = "Answer is grounded in the document."
        if result["off_topic"]["off_topic"]:
            off_topic_str = result["off_topic"]["reason"]
        
        report = "### 🛡️ Guardrails Report\n\n"
        report += f"**Verdict:** {verdict_str}\n\n"
        report += f"**PII Check:** {pii_str}\n\n"
        report += f"**Off-Topic Check:** {off_topic_str}"
        
        return report

    def safe_answer(self, answer: str, result: dict) -> str:
        verdict = result["verdict"]
        if verdict == "SAFE":
            return answer
        elif verdict == "OFF_TOPIC":
            return "⚠️ I can only answer questions about the uploaded document. This response was blocked."
        elif verdict == "PII_DETECTED":
            types = ", ".join(result["pii"]["types"])
            return f"🚨 This response was blocked because it contained sensitive personal information ({types})."
        else:
            types = ", ".join(result["pii"]["types"])
            return f"🚨 This response was blocked: off-topic and contained PII ({types})."
