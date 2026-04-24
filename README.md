# 📄 RAG Pipeline — Nava Software Assessment

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload a PDF and ask questions about it in a chat interface. Built with Groq's ultra-fast LLM inference, FAISS vector search, a cross-encoder reranker, and a Gradio UI.

---

## Features

- **PDF ingestion** — Upload any PDF through the UI or drop it in `data/`
- **Smart chunking** — Sliding window with sentence-boundary awareness
- **Text preprocessing** — Strips noise, collapses whitespace before indexing
- **Semantic search** — FAISS IndexFlatIP with L2-normalized embeddings (cosine similarity)
- **Cross-encoder reranking** — Rescores top-K retrieved chunks for precision
- **Conversational memory** — Rolling window (10 messages), auto-expires after 1 hour
- **Groq LLM** — `llama-3.3-70b-versatile` for fast, grounded answers

---

## 🗂️ Project Structure

```
NAVA/
├── .env                          # API keys and tunable parameters
├── config.py                     # Loads .env and exposes typed constants
├── main.py                       # Entry point — launches the Gradio app
├── requirements.txt
├── data/                         # Drop PDFs here (optional, can upload via UI)
├── app/
│   └── ui.py                     # Gradio layout, callbacks, global pipeline instance
└── src/
    ├── loaders/loader.py         # PDF text extraction (pypdf)
    ├── preprocessing/preprocess.py  # Text cleaning
    ├── chunking/chunker.py       # Sliding-window chunker
    ├── embeddings/embedder.py    # Sentence-transformer embedder (L2-normalized)
    ├── vectorstore/vector_store.py  # FAISS IndexFlatIP wrapper
    ├── retriever/retriever.py    # Embeds query → vector search
    ├── reranker/reranker.py      # Cross-encoder rescoring
    ├── memory/memory.py          # Thread-safe rolling chat memory with TTL
    ├── llm/llm.py                # Groq API client with RAG system prompt
    └── pipeline/rag_pipeline.py  # Orchestrates all components end-to-end
```

---

## 🧠 Architecture

```
PDF Upload
    │
    ▼
PDFLoader → Preprocessor → Chunker
                                │
                                ▼
                    Embedder (all-MiniLM-L6-v2)
                                │
                                ▼
                    VectorStore (FAISS IndexFlatIP)

User Query
    │
    ├─── Embedder → VectorStore.search (top 10)
    │                        │
    │                        ▼
    │               Reranker (cross-encoder, top 3)
    │                        │
    │                        ▼
    └──── ChatMemory.get_history() ──► LLM (Groq llama-3.3-70b)
                                              │
                                              ▼
                                         Answer + Memory update
```

---

## ⚙️ Tech Stack

| Component       | Library / Model                              |
|----------------|----------------------------------------------|
| LLM            | Groq API — `llama-3.3-70b-versatile`         |
| Embeddings     | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector DB      | `faiss-cpu` — IndexFlatIP                    |
| Reranker       | `cross-encoder/ms-marco-MiniLM-L-6-v2`       |
| UI             | `gradio`                                     |
| PDF Parsing    | `pypdf`                                      |
| Config         | `python-dotenv`                              |

---

## 🚀 Setup & Run

### 1. Clone / navigate to the project

```bash
cd /path/to/NAVA
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the embedding model (~90 MB) and the cross-encoder model (~80 MB) from Hugging Face. This is a one-time download.

### 4. Add your Groq API key

Open `.env` and replace the placeholder:

```env
GROQ_API_KEY=your_groq_api_key_here   # ← replace this
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
python main.py
```

Then open your browser at:

```
http://localhost:7860
```

---

## 🖥️ Using the UI

1. **Upload a PDF** using the file picker on the left panel.
2. Click **📤 Index File** — the pipeline loads, cleans, chunks, and embeds the document.
3. Type a question in the chat box on the right and press **Send ➤** or hit `Enter`.
4. The assistant answers using only the content from your PDF.
5. Click **🗑️ Reset All** to clear the index and conversation and start fresh.

---

## 🔧 Configuration

All parameters are controlled via `.env`:

| Variable               | Default                                    | Description                                  |
|------------------------|--------------------------------------------|----------------------------------------------|
| `GROQ_API_KEY`         | *(required)*                               | Your Groq API key                            |
| `GROQ_MODEL`           | `llama-3.3-70b-versatile`                  | Groq model to use                            |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`                         | Sentence-transformer model                   |
| `RERANKER_MODEL`       | `cross-encoder/ms-marco-MiniLM-L-6-v2`     | Cross-encoder model                          |
| `CHUNK_SIZE`           | `500`                                      | Max characters per chunk                     |
| `CHUNK_OVERLAP`        | `50`                                       | Overlap between consecutive chunks           |
| `TOP_K_RETRIEVE`       | `10`                                       | Chunks fetched from FAISS before reranking   |
| `TOP_K_RERANK`         | `3`                                        | Top chunks kept after reranking              |
| `MEMORY_MAX_MESSAGES`  | `10`                                       | Max conversation turns stored in memory      |
| `MEMORY_EXPIRY_SECONDS`| `3600`                                     | Session TTL in seconds (auto-reset on expiry)|

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ValueError: GROQ_API_KEY is missing` | Add your key to `.env` |
| `FileNotFoundError` on PDF | Make sure the file path is correct and the file exists |
| `ValueError: File must be a PDF` | Only `.pdf` files are supported |
| Port 7860 already in use | Change `server_port` in `main.py` |
| Slow first startup | Models are downloading for the first time — subsequent runs are instant |
