from src.llm import LLM

class HallucinationDetector:
    DETECTION_PROMPT = """You are a hallucination detection expert. Your job is to check whether an AI-generated answer is fully supported by the given context retrieved from a document.

Question: {question}
Retrieved Context: {context}
AI Answer: {answer}

Instructions:
1. Split the answer into individual claims.
2. For each claim, check if it is supported by the context.
3. Label each claim as:
   SUPPORTED — the context clearly backs this claim
   UNSUPPORTED — the context does not back this claim
   PARTIAL — the context partially backs this claim

Respond in this EXACT format:
CLAIM 1: <claim text> | <SUPPORTED/UNSUPPORTED/PARTIAL>
CLAIM 2: <claim text> | <SUPPORTED/UNSUPPORTED/PARTIAL>
...
CONFIDENCE: <0-100 integer representing % of claims that are SUPPORTED>
VERDICT: <GROUNDED if CONFIDENCE >= 70, HALLUCINATED otherwise>"""

    def __init__(self):
        self.llm = LLM()

    def detect(self, question: str, context: str, answer: str) -> dict:
        prompt = self.DETECTION_PROMPT.format(question=question, context=context, answer=answer)
        
        raw_response = self.llm.generate(user_message=prompt, context="", history=[])
        
        result = {
            "claims": [],
            "confidence": 0,
            "verdict": "ERROR",
            "raw": raw_response
        }

        try:
            lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
            for line in lines:
                if line.startswith("CLAIM"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        claim_data = parts[1].split("|")
                        if len(claim_data) == 2:
                            result["claims"].append({
                                "claim": claim_data[0].strip(),
                                "label": claim_data[1].strip()
                            })
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
                    # Strip any possible percentage signs or extra text
                    import re
                    match = re.search(r'\d+', confidence_str)
                    if match:
                        result["confidence"] = int(match.group())
                elif line.startswith("VERDICT:"):
                    verdict_str = line.replace("VERDICT:", "").strip()
                    result["verdict"] = verdict_str.split()[0] # Take the first word (GROUNDED/HALLUCINATED)

        except Exception:
            # On parse error, return the dictionary with verdict="ERROR" 
            pass

        return result

    def format_report(self, result: dict) -> str:
        if result["verdict"] == "ERROR":
            return f"### ⚠️ Hallucination Parse Error\n\n**Raw Output:**\n{result['raw']}"

        report = "### 🔍 Hallucination Detection Report\n\n"
        report += "| Claim | Status |\n|---|---|\n"
        
        for claim in result["claims"]:
            status_icon = "⚠️"
            if claim["label"] == "SUPPORTED":
                status_icon = "✅"
            elif claim["label"] == "UNSUPPORTED":
                status_icon = "❌"
            
            report += f"| {claim['claim']} | {status_icon} {claim['label']} |\n"

        verdict_icon = "✅" if "GROUNDED" in result["verdict"].upper() else "❌"
        
        report += f"\n**Confidence Score:** {result['confidence']}%\n"
        report += f"**Verdict:** {verdict_icon} {result['verdict']}"
        
        return report
