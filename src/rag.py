import os

from src.loader import PDFLoader
from src.chunker import Chunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.memory import ChatMemory
from src.llm import LLM


class RAGPipeline:
    def __init__(self):
        self.loader = PDFLoader()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.memory = ChatMemory()
        self.llm = LLM()
        self.current_file: str | None = None

    def load_pdf(self, filepath: str) -> str:
        """Index a new PDF. Replaces any previously loaded document."""
        self.vector_store.reset()
        self.memory.clear()

        text = self.loader.load(filepath)
        chunks = self.chunker.chunk(text)
        vectors = self.embedder.embed(chunks)
        self.vector_store.add(vectors, chunks)

        self.current_file = os.path.basename(filepath)
        return f"✅ Loaded '{self.current_file}' — {len(chunks)} chunks indexed."

    def query(self, user_message: str) -> str:
        """Answer a question using the indexed PDF content."""
        if self.vector_store.is_empty:
            return "⚠️ Please upload a PDF first."

        chunks = self.retriever.retrieve(user_message, top_k=5)
        context = "\n\n---\n\n".join(chunks)
        history = self.memory.get_history()

        answer = self.llm.generate(user_message, context, history)

        self.memory.add("user", user_message)
        self.memory.add("assistant", answer)

        return answer
