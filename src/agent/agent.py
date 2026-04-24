import string
from src.llm import LLM
from src.agent.tools import CalculatorTool, RAGSearchTool

class Agent:
    TOOL_SELECTION_PROMPT = """You are a routing agent. Given the user message below, decide which tool to use. Reply with ONLY the tool name — nothing else.

Available tools:
- calculator: for any math, arithmetic, or calculation questions
- rag_search: for any question about the uploaded document content

User message: {message}

Tool name:"""

    def __init__(self, pipeline):
        self.tools = {
            "calculator": CalculatorTool(),
            "rag_search": RAGSearchTool(pipeline)
        }
        self.llm = LLM()

    def run(self, user_message: str) -> dict:
        prompt = self.TOOL_SELECTION_PROMPT.format(message=user_message)
        
        # We pass empty context and history since this is a pure routing call
        response = self.llm.generate(user_message=prompt, context="", history=[])
        
        # Clean the output
        tool_name = response.strip().lower()
        tool_name = tool_name.strip(string.punctuation)

        # Fallback to rag_search if the LLM produces an invalid tool name
        if tool_name not in self.tools:
            tool_name = "rag_search"
            
        tool_result = self.tools[tool_name].run(user_message)
        
        return {
            "tool_used": tool_name,
            "result": tool_result
        }
