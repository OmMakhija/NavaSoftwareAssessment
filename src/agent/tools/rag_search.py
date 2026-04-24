class RAGSearchTool:
    name = "rag_search"
    description = "Searches the uploaded PDF document for relevant information. Use this for any question about the document content."

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, query: str) -> str:
        if self.pipeline.vector_store.is_empty:
            return "No document indexed. Please upload a PDF first."
        
        # The query method in the simplified pipeline directly returns the answer string
        answer = self.pipeline.query(query)
        return answer
