from .agents.researcher import ResearcherAgent
from .agents.summarizer import SummarizerAgent
from .agents.critic import CriticAgent

class Orchestrator:
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        
    def run(self, query: str, context: str) -> dict:
        research_facts = self.researcher.run(query, context)
        draft_summary = self.summarizer.run(query, research_facts)
        final_answer = self.critic.run(query, research_facts, draft_summary)
        
        return {
            "researcher": research_facts,
            "summarizer": draft_summary,
            "critic": final_answer,
            "final_answer": final_answer
        }
        
    def format_report(self, result: dict) -> str:
        report = "### 🤝 Multi-Agent Report\n\n"
        
        report += "#### 🔍 1. Researcher Extracted:\n"
        report += f"{result['researcher']}\n\n---\n\n"
        
        report += "#### ✍️ 2. Summarizer Drafted:\n"
        report += f"{result['summarizer']}\n\n---\n\n"
        
        report += "#### ⚖️ 3. Critic Finalized:\n"
        report += f"{result['critic']}"
        
        return report
