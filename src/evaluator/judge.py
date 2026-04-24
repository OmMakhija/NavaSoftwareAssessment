from src.llm import LLM

class Judge:
    JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an AI-generated answer to a question, based on a given context retrieved from a document.

Question: {question}

Retrieved Context: {context}

AI Answer: {answer}

Evaluate the answer on these three criteria. For each, give:
- A score from 1 to 5 (1=very poor, 5=excellent)
- One sentence of reasoning

Criteria:
1. Relevance: Is the answer relevant to the question?
2. Faithfulness: Is the answer grounded in the retrieved context and free from hallucinations?
3. Completeness: Does the answer fully address the question?

Respond in this EXACT format, nothing else:
RELEVANCE: <score>/5 | <reasoning>
FAITHFULNESS: <score>/5 | <reasoning>
COMPLETENESS: <score>/5 | <reasoning>
VERDICT: <PASS if average >= 3.5, FAIL otherwise> | <one line summary>"""

    def __init__(self):
        self.llm = LLM()

    def evaluate(self, question: str, context: str, answer: str) -> dict:
        prompt = self.JUDGE_PROMPT.format(question=question, context=context, answer=answer)
        
        raw_response = self.llm.generate(user_message=prompt, context="", history=[])
        
        result = {
            "relevance": {"score": 0, "reasoning": ""},
            "faithfulness": {"score": 0, "reasoning": ""},
            "completeness": {"score": 0, "reasoning": ""},
            "average_score": 0.0,
            "verdict": "ERROR",
            "summary": "",
            "raw": raw_response
        }

        try:
            lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
            for line in lines:
                if line.startswith("RELEVANCE:"):
                    parts = line.replace("RELEVANCE:", "").split("|")
                    result["relevance"]["score"] = int(parts[0].strip().split("/")[0])
                    result["relevance"]["reasoning"] = parts[1].strip()
                elif line.startswith("FAITHFULNESS:"):
                    parts = line.replace("FAITHFULNESS:", "").split("|")
                    result["faithfulness"]["score"] = int(parts[0].strip().split("/")[0])
                    result["faithfulness"]["reasoning"] = parts[1].strip()
                elif line.startswith("COMPLETENESS:"):
                    parts = line.replace("COMPLETENESS:", "").split("|")
                    result["completeness"]["score"] = int(parts[0].strip().split("/")[0])
                    result["completeness"]["reasoning"] = parts[1].strip()
                elif line.startswith("VERDICT:"):
                    parts = line.replace("VERDICT:", "").split("|")
                    result["verdict"] = parts[0].strip()
                    result["summary"] = parts[1].strip()

            scores = [
                result["relevance"]["score"], 
                result["faithfulness"]["score"], 
                result["completeness"]["score"]
            ]
            
            result["average_score"] = round(sum(scores) / len(scores), 2)
            
        except Exception:
            # On parse error, the result dictionary will have verdict="ERROR" 
            # and raw_response will be accessible.
            pass

        return result

    def format_report(self, evaluation: dict) -> str:
        if evaluation["verdict"] == "ERROR":
            return f"### ⚠️ Evaluation Parse Error\n\n**Raw Output:**\n{evaluation['raw']}"

        verdict_icon = "✅" if "PASS" in evaluation["verdict"].upper() else "❌"
        
        return f"""### 📊 Evaluation Report
| Criterion | Score | Reasoning |
|---|---|---|
| Relevance | {evaluation['relevance']['score']}/5 | {evaluation['relevance']['reasoning']} |
| Faithfulness | {evaluation['faithfulness']['score']}/5 | {evaluation['faithfulness']['reasoning']} |
| Completeness | {evaluation['completeness']['score']}/5 | {evaluation['completeness']['reasoning']} |

**Average Score:** {evaluation['average_score']}/5
**Verdict:** {verdict_icon} {evaluation['verdict']}
**Summary:** {evaluation['summary']}"""
