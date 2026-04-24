class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        chunks = []
        i = 0

        while i < len(text):
            end = min(i + self.chunk_size, len(text))

            # Try to break at a sentence boundary near the end of the window
            if end < len(text):
                boundary = text.rfind(". ", max(i + int(self.chunk_size * 0.7), i), end)
                if boundary != -1:
                    end = boundary + 2

            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            next_i = end - self.chunk_overlap
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return chunks
