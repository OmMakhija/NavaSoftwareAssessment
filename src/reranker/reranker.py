from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model = CrossEncoder(model_name)
        print(f"Reranker initialized with model: {model_name}")

    def rerank(self, query: str, results: list[dict], top_k: int = 3) -> list[dict]:
        if not results:
            return []

        # Score each result using the cross-encoder
        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        # Sort by rerank_score descending
        reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:top_k]

    def format_comparison(self, original: list[dict], reranked: list[dict]) -> str:
        report = "### 🔀 Re-ranker Report\n\n"
        
        report += "**Before Re-ranking (vector similarity order):**\n"
        report += "| Rank | Score | Preview |\n|---|---|---|\n"
        for i, r in enumerate(original):
            preview = r["text"][:80].replace("\n", " ") + "..."
            score = round(r.get("score", 0.0), 4)
            report += f"| {i+1} | {score} | {preview} |\n"

        report += "\n**After Re-ranking (cross-encoder order):**\n"
        report += "| Rank | Score | Preview |\n|---|---|---|\n"
        for i, r in enumerate(reranked):
            preview = r["text"][:80].replace("\n", " ") + "..."
            score = round(r.get("rerank_score", 0.0), 4)
            report += f"| {i+1} | {score} | {preview} |\n"

        return report
