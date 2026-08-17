

https://github.com/user-attachments/assets/700123d7-f2cc-404b-bc37-e19480215a65

# 🤖 Local RAG Knowledge Assistant (Microsoft Foundry Local & SQLite)

A modular, high-performance, and **100% offline** Local Retrieval-Augmented Generation (RAG) Knowledge Assistant built with Python, Microsoft Foundry Local (`foundry-local-sdk`), SQLite3, NumPy, and Streamlit.

---

## 🌟 Overview

This application allows users to query local documents (`.txt` and `.md`) completely offline with **zero cloud API dependencies, zero external network calls, and complete data privacy**.

### Key Highlights & Workflow:
1. **Document Ingestion & Contextual Chunking**: Text files in `documents/` are segmented into overlapping chunks (~500 chars with ~50 char overlap) respecting natural sentence and word boundaries.
2. **Topical Disambiguation**: Embeddings are computed with document-level topic prefixes (e.g., `Konu: süper lig\n\n...`) to eliminate semantic confusion across similar topics.
3. **Local Vector Embeddings**: Chunks are embedded on-device using Microsoft Foundry Local's embedding model (e.g., `qwen3-embedding-0.6b`).
4. **SQLite Vector Storage**: Document text, chunk IDs, file metadata, and vector representations are persisted in a local `rag_knowledge.db` SQLite database.
5. **Vector Retrieval & Filtering**: User queries are embedded and compared against stored vectors using NumPy cosine similarity with a relevance threshold (`min_score = 0.20`) and Top-$K$ selection.
6. **Strict Grounded Generation**: Retrieved passages are injected into a strict system prompt and processed by a local LLM via Microsoft Foundry Local (`phi-3.5-mini`) to generate accurate, hallucination-free, source-cited responses in real time with token streaming.

---

## 📁 Project Structure

```text
rag-foundry-assistant/
│
├── documents/                       # Knowledge base documents (.txt, .md)
│   ├── catenaccio.txt               # Catenaccio tactical system & history
│   ├── fifa_dünya_kupası.txt        # FIFA World Cup history & statistics
│   ├── futbol.txt                   # General football rules, origins & history
│   ├── futbol_kuralları.txt         # Official rules (IFAB, match dynamics)
│   ├── ofsayt.txt                   # Offside rule explanations & exceptions
│   ├── süper_lig.txt                # Turkish Süper Lig champions & records
│   ├── uefa_avrupa_ligi.txt         # UEFA Europa League history & winners
│   └── uefa_şampiyonlar_ligi.txt    # UEFA Champions League titles & records
│
├── database.py                      # SQLite schema, batch insertion & vector retrieval queries
├── ingestion.py                     # Document parser, smart text chunking & embedding pipeline
├── rag_engine.py                    # Cosine similarity math, prompt formatting & LLM inference
├── app.py                           # Streamlit interactive web interface with real-time streaming
├── cli.py                           # Interactive terminal CLI alternative
├── rag_knowledge.db                 # Local SQLite database storing text chunks & vector embeddings
├── requirements.txt                 # Python dependencies
├── video/                           # Application demo recording
│   └── RAG video.mp4                # Demo video file
└── README.md                        # Documentation & setup guide
```

---

## 🎥 Demo Video

> Projenin uçtan uca çalışmasını, doküman indeksleme sürecini ve yerel modelin anlık yanıtlarını izlemek için:
> - Repodaki [`video/RAG video.mp4`](video/RAG%20video.mp4) dosyasını doğrudan GitHub üzerinde açıp oynatabilirsiniz.

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment
Requires **Python 3.10+** (Tested on Python 3.11 / 3.13).

```bash
cd rag-foundry-assistant
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (CMD):
.\venv\Scripts\activate.bat

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for Hardware Acceleration (Optional)**: If running on Windows with DirectX / WinML support:
> ```bash
> pip install foundry-local-sdk-winml
> ```

---

## 🚀 How to Run the Application

### Option A: Interactive Streamlit Web UI (Recommended)

Launch the Streamlit web application:
```bash
streamlit run app.py
```

- Open `http://localhost:8501` in your browser.
- **Sidebar Actions**:
  - Check the database status and see currently indexed files.
  - Configure LLM alias (`phi-3.5-mini`), Embedding alias (`qwen3-embedding-0.6b`), and Top-$K$ retrieval slider.
  - Click **"🚀 Process & Ingest Documents into SQLite"** to index/re-index documents.
- **Chat Interface**:
  - Ask questions in Turkish or English regarding your indexed documents.
  - Real-time token streaming responses.
  - Expand **"🔍 View Retrieved Context Chunks & Similarity Scores"** under any answer to inspect matched source files, chunk IDs, and cosine similarity scores.

---

### Option B: Command Line Interface (CLI Alternative)

For headless or terminal-only environments:
```bash
python cli.py
```

```text
MAIN MENU:
  [1] Process & Ingest Documents ('documents/' -> SQLite)
  [2] Ask a Question (Interactive Q&A)
  [3] Exit CLI
```

- Select `[1]` to parse and embed all files inside `documents/`.
- Select `[2]` to enter the interactive Q&A session with passage snippets and similarity score breakdowns.
- Select `[3]` to exit.

---

## ⚙️ Technical Architecture & Features

### 1. SQLite Database Layer (`database.py`)
Stores text passages and high-dimensional vector embeddings in `rag_knowledge.db` under the `documents_chunks` table:
* `id` (`INTEGER PRIMARY KEY AUTOINCREMENT`)
* `file_name` (`TEXT NOT NULL` with index `idx_file_name`)
* `chunk_id` (`INTEGER NOT NULL`)
* `content` (`TEXT NOT NULL`)
* `embedding` (`TEXT NOT NULL` — JSON-serialized float array)

### 2. Ingestion & Embedding Pipeline (`ingestion.py`)
* **Natural Boundary Chunking**: Splits text (~500 chars with ~50 char overlap) snapping forward to word and sentence boundaries to avoid splitting words.
* **Document-Level Contextual Embeddings**: Prepends a derived topic label to the embedding input (`build_embedding_input`) so distinct tournaments (e.g. Süper Lig vs. UEFA Champions League) maintain clean separation in vector space.
* **Batch Insertion**: Optimizes database writes using transactional `executemany`.

### 3. Retrieval & RAG Engine (`rag_engine.py`)
* **Cosine Similarity Calculation**:
  $$\text{Cosine Similarity}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|_2 \cdot \|\vec{b}\|_2}$$
* **Threshold Filtering**: Discards irrelevant noise below `min_score = 0.20`.
* **Strict Grounded System Prompt**:
  Forces the model to strictly rely on the provided context passages, accurately distinguish between distinct entities/tournaments, provide source citations (`Kaynak: dosya_adi.txt`), and answer *"Bu bilgi belgelerimde bulunmuyor."* when information is absent.
* **Streaming & Non-Streaming Inference**: Full support for both `complete_streaming_chat` (used in Streamlit) and standard synchronous completions.

---

## 🔒 Offline & Privacy Guarantees

* **100% On-Device**: All model weights, inference execution, and vector calculations run locally through Microsoft Foundry Local ONNX Runtime.
* **Zero Telemetry/Cloud Leaks**: Data never leaves your machine.
* **Local Persistence**: Document chunks and embeddings are stored locally in the standard SQLite database file.

---

## 🐞 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **"Bu bilgi belgelerimde bulunmuyor."** | Ensure you have clicked the **"Process & Ingest Documents"** button in the UI or run Option `[1]` in `cli.py` to index the documents. |
| **Model download on first startup** | `foundry-local-sdk` downloads and caches the requested models locally upon first invocation. Subsequent runs are instant and fully offline. |
| **Port 8501 is in use** | Run `streamlit run app.py --server.port 8502` to use a different port. |
