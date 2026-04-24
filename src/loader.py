import os
from pypdf import PdfReader


class PDFLoader:
    def load(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        if not filepath.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        reader = PdfReader(filepath)
        pages = []

        for i, page in enumerate(reader.pages):
            # Strategy 1: standard extraction
            text = page.extract_text()

            # Strategy 2: fallback to layout mode if standard returns nothing
            if not text or not text.strip():
                try:
                    text = page.extract_text(extraction_mode="layout")
                except Exception:
                    text = ""

            if text and text.strip():
                pages.append(text.strip())

        full_text = "\n\n".join(pages)

        if not full_text.strip():
            print(
                "[WARNING] PDF appears to contain no extractable text. "
                "It may be a scanned image or image-based PDF that requires OCR. "
                "Returning empty string — please use an OCR tool to pre-process it."
            )
            return ""

        filename = os.path.basename(filepath)
        print(f"Loaded '{filename}': {len(full_text)} characters across {len(pages)} page(s).")

        return full_text
