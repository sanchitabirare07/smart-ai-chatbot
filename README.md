# 🤖 SmartBot — AI Chatbot for College / Customer Support

A production-ready **RAG (Retrieval Augmented Generation)** chatbot built with:
- **LangChain** for orchestration
- **FAISS** for vector storage
- **OpenAI / Gemini** as LLM backends
- **Streamlit** for the web UI

---

## 📁 Project Structure

```
smart-ai-chatbot/
├── app/
│   └── main.py              # Streamlit UI entry point
├── utils/
│   ├── __init__.py
│   ├── rag_pipeline.py      # Core RAG logic
│   └── chat_history.py      # Chat session management
├── data/
│   └── sample_faq.txt       # Sample knowledge base document
├── vectorstore/             # Auto-created: FAISS index files
├── .streamlit/
│   └── config.toml          # Streamlit theme config
├── .env.example             # Copy to .env and add your API keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone / Open the project
```bash
cd smart-ai-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Open .env and paste your OpenAI or Gemini API key
```

### 5. Run the app
```bash
streamlit run app/main.py
```

---

## 🚀 How to Use

1. **Open** the app in your browser (usually http://localhost:8501)
2. **Sidebar** → Select AI Provider (OpenAI or Gemini) and paste your API key
3. **Upload** a PDF or TXT file (use `data/sample_faq.txt` to test)
4. Click **"Build Knowledge Base"** — this creates the FAISS vector store
5. **Ask questions** in the chat input or click quick FAQ buttons
6. The bot retrieves relevant context from your documents and generates answers

---

## 🔑 Getting API Keys

| Provider | URL |
|----------|-----|
| OpenAI   | https://platform.openai.com/api-keys |
| Gemini   | https://aistudio.google.com/app/apikey |

---

## 🛠️ VS Code Extensions to Install

See the section below for the full list.

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| **RAG** | Retrieval Augmented Generation — retrieves relevant chunks before generating answers |
| **FAISS** | Facebook AI Similarity Search — fast nearest-neighbour vector search |
| **Embeddings** | Text → numeric vectors that capture semantic meaning |
| **Chunking** | Splitting documents into overlapping pieces for better retrieval |
| **LangChain** | Framework to chain LLMs, retrievers, prompts together |

---

## 🧩 Extending the Project

- **Add ChromaDB**: Replace `FAISS` with `Chroma` in `rag_pipeline.py` for persistent storage
- **Add memory**: Use `ConversationBufferMemory` in LangChain for multi-turn awareness
- **Deploy**: Push to Streamlit Cloud (`streamlit.io`) — free hosting for Streamlit apps
- **Add more loaders**: Support DOCX, CSV, URLs via LangChain's document loaders
