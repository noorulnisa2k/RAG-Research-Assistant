# 🔍 Multi-Agent RAG Research Assistant

A **Streamlit-based Retrieval-Augmented Generation (RAG) system** that combines intelligent document retrieval with multi-agent AI for synthesizing research answers. Upload your research papers, PDFs, or text files, and let the system retrieve relevant content and generate well-cited answers to your questions.

## ✨ Features

- **📄 Document Upload**: Upload PDF and TXT files directly via the UI
- **🔍 Smart Retrieval**: Uses ChromaDB + Sentence Transformers for semantic search
- **🤖 Multi-Agent Pipeline**: 
  - **PlannerAgent**: Decomposes complex queries into sub-questions
  - **RetrieverAgent**: Fetches relevant document chunks
  - **AnswerAgent**: Synthesizes structured answers with citations
- **📚 Citation Management**: Automatic APA-style inline citations
- **💾 Persistent Storage**: ChromaDB stores embeddings for fast retrieval
- **⚡ Local LLM Support**: Works with local LLMs (LM Studio, Ollama, etc.)

## 🛠️ Setup

### 1. Prerequisites

- Python 3.8+
- Local LLM server running (e.g., LM Studio, Ollama)
- Internet connection (first run downloads sentence-transformers model)

### 2. Installation

```bash
# Clone or download the app
cd your-project-folder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# LLM Configuration (LM Studio)
LLAMA_MODEL=Llama-3.2-3B-Instruct
BASE_URL=http://localhost:1234/v1
API_KEY=lm-studio

# Embedding Model (local, no API key)
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**For different LLM providers:**

- **LM Studio** (Recommended for beginners):
  ```
  LLAMA_MODEL=Llama-3.2-3B-Instruct
  BASE_URL=http://localhost:1234/v1
  API_KEY=lm-studio
  ```

- **Ollama**:
  ```
  LLAMA_MODEL=mistral
  BASE_URL=http://localhost:11434/v1
  API_KEY=ollama
  ```

- **OpenAI API** (if you have a key):
  ```
  LLAMA_MODEL=gpt-3.5-turbo
  BASE_URL=https://api.openai.com/v1
  API_KEY=your-openai-key
  ```

### 4. Start Your Local LLM Server

**Using LM Studio:**
1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load a model (e.g., Llama 3.2 3B)
3. Start the local server on port 1234
4. Verify: `curl http://localhost:1234/v1/models`

**Using Ollama:**
```bash
ollama pull mistral
ollama serve
```

### 5. Run the App

```bash
streamlit run streamlit_rag_app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage

### 1. Initialize Database
- Click **"🔄 Initialize/Reset Database"** to set up ChromaDB

### 2. Upload Documents
- Use the **"📄 Upload Documents"** section to upload PDF or TXT files
- Click **"Process Documents"** to extract text and create embeddings

### 3. Ask a Question
- Enter your research question in the text area
- Click **"🚀 Ask Research Assistant"**
- System will:
  1. Retrieve relevant document chunks
  2. Plan the research approach
  3. Synthesize a structured, cited answer

### 4. Review Results
- **Retrieved Sources**: See which documents were used
- **Multi-Agent Response**: Fully structured answer with citations
- **References Section**: All sources cited in the response

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI                          │
│  (File Upload + Query Input + Results Display)         │
└────────────────┬────────────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
    ┌────▼──────┐    ┌───▼──────────┐
    │  ChromaDB  │    │ Sentence-    │
    │ (Vector DB)│    │ Transformers │
    │            │    │ (Embeddings) │
    └────────────┘    └──────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   AutoGen Multi-Agent Pipeline    │
    │  ┌──────────┐ ┌──────────┐        │
    │  │ Planner  │ │Retriever │        │
    │  │  Agent   │ │  Agent   │        │
    │  └──────────┘ └──────────┘        │
    │  ┌──────────┐ ┌──────────┐        │
    │  │  Answer  │ │   User   │        │
    │  │  Agent   │ │  Proxy   │        │
    │  └──────────┘ └──────────┘        │
    └────────────────────────────────────┘
         │
    ┌────▼────────────────┐
    │   Local LLM Server   │
    │ (Llama, Mistral, etc)│
    └─────────────────────┘
```

## 📊 Configuration

### Sidebar Controls

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Top K Results | 1-10 | 5 | Number of document chunks to retrieve |
| Temperature | 0.0-1.0 | 0.3 | LLM creativity (lower = more deterministic) |

### Document Processing

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk Size | 500 | Characters per chunk |
| Chunk Overlap | 100 | Overlap between consecutive chunks |
| Embedding Model | all-MiniLM-L6-v2 | Fast, local, no API key |

## 🧠 How It Works

### 1. **Document Ingestion**
   - Upload PDFs/TXTs → Extract text → Split into chunks
   - Create embeddings using Sentence Transformers

### 2. **Indexing**
   - Store chunks + embeddings in ChromaDB
   - Enable fast semantic search

### 3. **Query Processing**
   - User enters research question
   - PlannerAgent decomposes into sub-queries
   - RetrieverAgent retrieves relevant chunks
   - AnswerAgent synthesizes citations answer

### 4. **Response Generation**
   - Structured markdown output
   - Inline APA-style citations
   - References section with sources

## 🔧 Troubleshooting

### Issue: "Connection refused" error
**Solution**: Make sure your local LLM server is running
```bash
# Check if server is running
curl http://localhost:1234/v1/models

# If not, restart LM Studio or Ollama
```

### Issue: "No module named 'chromadb'"
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt
```

### Issue: Slow response time
**Solutions**:
- Reduce `Top K Results` (sidebar)
- Use a smaller LLM model
- Reduce `Temperature` for faster generation
- Check your local LLM's VRAM usage

### Issue: Low-quality answers
**Solutions**:
- Upload more relevant documents
- Rephrase your query more specifically
- Check document quality (OCR for scanned PDFs)
- Increase `Top K Results`

## 📝 Example Queries

Here are some queries you can try after uploading research papers:

- "What is Automated Essay Scoring and how do modern LLMs improve upon traditional approaches?"
- "Compare transformer-based models vs classical ML for essay evaluation"
- "What are the main challenges in writing assessment systems?"
- "How does RAG compare to fine-tuning for domain-specific tasks?"
- "Explain the multi-agent architecture used in this system"

## 🚀 Performance Tips

1. **Start Small**: Upload 3-5 documents first to test
2. **Optimize Chunks**: Increase overlap if answers are fragmented
3. **Monitor Resources**: Check GPU/CPU while processing
4. **Cache Results**: ChromaDB persists, no re-indexing needed
5. **Parallel Processing**: Process multiple documents at once

## 📦 File Structure

```
project-root/
├── streamlit_rag_app.py      # Main Streamlit app
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── chroma_db/                 # ChromaDB persistence (auto-created)
└── README.md                  # This file
```

## 🎯 Future Enhancements

- [ ] Web search integration for real-time data
- [ ] Multi-language support
- [ ] Custom citation formats (MLA, Chicago, Harvard)
- [ ] Export to PDF/Word
- [ ] Query history and bookmarking
- [ ] Advanced filtering by source/date
- [ ] Interactive citation editing
- [ ] Performance metrics dashboard

## 📄 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section
2. Verify your `.env` configuration
3. Check local LLM server logs
4. Open an issue with:
   - Error message
   - Steps to reproduce
   - Your environment (OS, Python version, etc.)

---

**Built with ❤️ using Streamlit, ChromaDB, AutoGen, and local LLMs**
