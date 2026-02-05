# 🚀 YTRAG - Retrieval-Augmented Generation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2.7+-00A3E0?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B6B?style=for-the-badge)
![Google Generative AI](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0096d6?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-00C851?style=for-the-badge)

**A Powerful Retrieval-Augmented Generation System for Document Intelligence & Question Answering**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Pipeline Workflow](#pipeline-workflow)

</div>

---

## 📋 Overview

**YTRAG** is an enterprise-grade **Retrieval-Augmented Generation (RAG)** system that combines state-of-the-art document processing, semantic search, and generative AI to create intelligent question-answering systems. It enables you to ingest multiple document types (PDFs, text files, CSV), convert them to embeddings, store them securely in a vector database, and retrieve relevant context for precise AI-generated answers.

### 🎯 Key Capabilities
- **Multi-format Document Support**: Process PDFs, Text files, and CSV data
- **Semantic Search**: Find relevant documents using embedding similarity
- **LLM Integration**: Powered by Google Gemini for intelligent responses  
- **Vector Storage**: Persistent ChromaDB for efficient retrieval
- **Advanced Retrieval**: Confidence scoring, source tracking, and history management
- **Streaming & Summarization**: Real-time responses and content summarization

---

## ✨ Features

### 📁 Data Ingestion Pipeline
- ✅ **TextLoader**: Load individual text files with custom encoding
- ✅ **DirectoryLoader**: Batch load all files from a directory
- ✅ **PyPDFLoader/PyMuPDFLoader**: Extract text and metadata from PDFs
- ✅ **CSVLoader**: Process structured data in CSV format
- ✅ **Metadata Preservation**: Maintain source, page, author information

### 🔤 Document Processing
- ✅ **RecursiveCharacterTextSplitter**: Intelligent chunking with overlap
- ✅ **Configurable Chunk Sizes**: Customize for your use case (1000 chars default)
- ✅ **Context Preservation**: Overlapping chunks maintain semantic continuity
- ✅ **Smart Separators**: Hierarchical split strategy (\n\n → \n → space → char)

### 🧠 Embedding Generation
- ✅ **SentenceTransformer**: Using `all-MiniLM-L6-v2` model
- ✅ **High-Dimensional Embeddings**: 384-dimensional vector representations
- ✅ **Batch Processing**: Efficient encoding with progress tracking
- ✅ **Semantic Understanding**: Capture meaning beyond keywords

### 🗂️ Vector Storage & Retrieval
- ✅ **ChromaDB Integration**: Persistent vector database
- ✅ **Similarity Search**: Cosine distance-based retrieval
- ✅ **Metadata Indexing**: Filter and track document sources
- ✅ **Scalable Storage**: Handle thousands of documents efficiently

### 🤖 LLM-Powered Generation
- ✅ **Google Gemini 2.5 Flash**: Fast, accurate responses
- ✅ **Prompt Engineering**: Optimized context-aware prompts
- ✅ **Temperature Control**: Adjustable response creativity (0-2)
- ✅ **Token Management**: Control output length (up to 1024 tokens)

### 🔍 Advanced RAG Features
- ✅ **Dual Retrieval Modes**: Simple RAG + Advanced RAG with citations
- ✅ **Confidence Scoring**: Know how relevant your results are
- ✅ **Source Attribution**: Track which documents powered each answer
- ✅ **Session History**: Maintain conversation context
- ✅ **Streaming Output**: Real-time response generation
- ✅ **Summarization**: Auto-generate concise summaries

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   1. QUERY EMBEDDING        │
        │  (SentenceTransformer)      │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  2. SIMILARITY SEARCH       │
        │   (ChromaDB/FAISS)          │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │  3. CONTEXT RETRIEVAL & RANKING     │
        │   - Top-K Results                   │
        │   - Filter by Confidence            │
        │   - Prepare Sources                 │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │  4. PROMPT CONSTRUCTION             │
        │   - Context Injection               │
        │   - Query Integration               │
        │   - Format Optimization             │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │  5. LLM GENERATION (Gemini)         │
        │   - Stream Response                 │
        │   - Temperature: 0.1 (Factual)      │
        │   - Max Tokens: 1024                │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │  6. OUTPUT ENRICHMENT               │
        │   - Add Confidence Score            │
        │   - Attach Sources                  │
        │   - Store History                   │
        │   - Optional Summarization          │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   ANSWER + METADATA         │
        │   + Citations + Confidence  │
        └─────────────────────────────┘
```

---

## 🔄 Pipeline Workflow

### Phase 1: Data Ingestion & Processing

```
Multiple Sources
  ├─ PDF Files (PyMuPDFLoader)
  ├─ Text Files (TextLoader)
  └─ CSV Files (CSVLoader)
         │
         ▼
  DirectoryLoader (Batch Processing)
         │
         ▼
  Langchain Documents (page_content + metadata)
         │
         ▼
  RecursiveCharacterTextSplitter
  ├─ Chunk Size: 1000 characters
  ├─ Overlap: 200 characters
  └─ Hierarchical separators
         │
         ▼
  Document Chunks (Ready for Embedding)
```

### Phase 2: Embedding & Vector Storage

```
Text Chunks
     │
     ▼
SentenceTransformer (all-MiniLM-L6-v2)
     │
     ├─ 384-dimensional vectors
     ├─ Semantic representation
     └─ Ready for similarity search
     │
     ▼
ChromaDB Collection
     ├─ Store embeddings
     ├─ Store metadata
     ├─ Persist to disk
     └─ Enable fast retrieval
```

### Phase 3: Query Processing & Retrieval

```
User Query
     │
     ▼
Embedding Generation (Same Model)
     │
     ▼
Vector Similarity Search (Cosine Distance)
     │
     ├─ Top-K retrieval (configurable)
     ├─ Score threshold filtering
     └─ Similarity scoring
     │
     ▼
Retrieved Documents + Rankings
     │
     ├─ Document content
     ├─ Metadata (source, page, author)
     ├─ Confidence scores
     └─ Source attribution
```

### Phase 4: Answer Generation

```
Retrieved Context
     │
     ├─ Combine with user query
     ├─ Inject into prompt template
     └─ Format for LLM
     │
     ▼
Google Gemini 2.5 Flash
     │
     ├─ Temperature: 0.1 (Factual)
     ├─ Max tokens: 1024
     └─ Stream output (optional)
     │
     ▼
Generated Answer + Sources + Confidence Score
```

---

## 🛠️ Installation

### Prerequisites
- **Python 3.13+**
- **pip** or **uv** package manager
- **Google Generative AI API Key**

### Step 1: Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/YTRAG.git
cd YTRAG

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows
```

### Step 2: Install Dependencies

Using `pip`:
```bash
pip install -r requirements.txt
```

Or using `uv` (faster):
```bash
uv add -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_generative_ai_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 1.2.7+ | Core RAG framework |
| `langchain-core` | 1.2.7+ | Base components |
| `langchain-community` | 0.4.1+ | Document loaders |
| `langchain-google-genai` | 4.2.0+ | Gemini LLM integration |
| `sentence-transformers` | 5.2.2+ | Embedding generation |
| `chromadb` | 1.4.1+ | Vector database |
| `faiss-cpu` | 1.13.2+ | Vector similarity search |
| `pypdf` | 6.6.2+ | PDF processing |
| `pymupdf` | 1.26.7+ | Advanced PDF extraction |
| `python-dotenv` | Latest | Environment variables |
| `google-generativeai` | 0.8.6+ | Gemini API client |

---

## 🚀 Usage

### Basic Usage: Simple RAG

```python
from notebook.document import (
    TextLoader, DirectoryLoader, 
    RecursiveCharacterTextSplitter,
    EmbeddingManager, VectorStore, RAGRetriever,
    ChatGoogleGenerativeAI, rag_simple
)
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# 1. Load documents
dir_loader = DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = dir_loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Generate embeddings
embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2')
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)

# 4. Store in vector database
vector_store = VectorStore(collection_name="documents")
vector_store.add_documents(chunks, embeddings)

# 5. Create retriever
rag_retriever = RAGRetriever(vector_store, embedding_manager)

# 6. Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=0.1
)

# 7. Query with Simple RAG
answer = rag_simple("Your question here?", rag_retriever, llm, top_k=3)
print(answer)
```

### Advanced Usage: Enhanced RAG with Citations

```python
# Use the advanced RAG pipeline
result = rag_advanced(
    query="What are Industrial Designs?",
    retriever=rag_retriever,
    llm=llm,
    top_k=5,
    min_score=0.2,
    return_context=True
)

print("Answer:", result["answer"])
print("Confidence:", result["confidence"])
print("Sources:", result["sources"])
print("Context:", result["context"][:300])
```

### Enterprise Usage: Full Advanced Pipeline

```python
# Initialize advanced pipeline
adv_rag = AdvancedRAGPipeline(rag_retriever, llm)

# Query with streaming, summarization, and history
result = adv_rag.query(
    question="APPLE INC. VS. SAMSUNG case summary",
    top_k=5,
    min_score=0.1,
    stream=True,        # Stream response
    summarize=True      # Auto-summarize
)

print("Answer:", result["answer"])
print("Summary:", result["summary"])
print("History:", result["history"])
print("Sources:", result["sources"])
```

---

## 📊 Data Structure

### Document Metadata Example
```python
{
    "source": "file1.txt",
    "page": 1,
    "author": "Akhil Shibu",
    "date_created": "2026-02-02",
    "content_length": 1250,
    "doc_index": 0
}
```

### Retrieval Result Structure
```python
{
    "id": "doc_a1b2c3d4_0",
    "content": "Text chunk content...",
    "metadata": {...},  # See above
    "similarity_score": 0.87,
    "distance": 0.13,
    "rank": 1
}
```

### RAG Query Response
```python
{
    "answer": "Generated answer...",
    "sources": [
        {
            "source": "file1.txt",
            "page": 1,
            "score": 0.87,
            "preview": "Text preview..."
        }
    ],
    "confidence": 0.87,
    "summary": "Concise answer summary...",
    "history": [...]  # Previous queries
}
```

---

## 📂 Project Structure

```
YTRAG/
├── main.py                 # Entry point
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── README.md              # This file
├── .env                   # API keys (create this)
│
├── data/
│   ├── text_files/        # Input text documents
│   │   ├── file1.txt
│   │   └── file2.txt
│   ├── pdf/               # Input PDF documents
│   ├── Csv_files/         # Input CSV documents
│   │   └── rag_langchain_dataset.csv
│   └── vector_store/      # ChromaDB persistence
│       └── chroma.sqlite3
│
└── notebook/
    └── document.ipynb     # Main Jupyter notebook with all pipelines
```

---

## 🔑 Key Classes

### 1. **EmbeddingManager**
Handles embedding generation using SentenceTransformers
```python
embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2')
embeddings = embedding_manager.generate_embeddings(texts)
```

### 2. **VectorStore**
Manages document embeddings using ChromaDB
```python
vector_store = VectorStore(collection_name="documents", persist_directory="../data/vector_store")
vector_store.add_documents(documents, embeddings)
```

### 3. **RAGRetriever**
Retrieves relevant documents based on query similarity
```python
rag_retriever = RAGRetriever(vector_store, embedding_manager)
results = rag_retriever.retrieve(query, top_k=5, score_threshold=0.2)
```

### 4. **AdvancedRAGPipeline**
Enterprise-grade RAG with history, streaming, and summarization
```python
adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
result = adv_rag.query(question, top_k=5, stream=True, summarize=True)
```

---

## ⚙️ Configuration

### Customize Chunk Splitting
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # Larger chunks for summary
    chunk_overlap=400   # More overlap for context
)
```

### Adjust Embedding Model
```python
embedding_manager = EmbeddingManager(
    model_name='all-mpnet-base-v2'  # Better but slower
)
```

### Fine-tune LLM
```python
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=0.5,        # More creative
    max_output_tokens=2048  # Longer answers
)
```

### Retrieval Parameters
```python
# More results, higher threshold
results = rag_retriever.retrieve(
    query=question,
    top_k=10,              # Get top 10
    score_threshold=0.5    # Only very similar docs
)
```

---

## 🧪 Testing & Examples

### Test Query 1: Simple Question
```python
answer = rag_simple(
    "What are the acts and laws governing Industrial Design?",
    rag_retriever, llm, top_k=3
)
```

### Test Query 2: Complex Case Analysis  
```python
result = rag_advanced(
    "APPLE INC. VS. SAMSUNG ELECTRONICS case summary",
    rag_retriever, llm, top_k=5, return_context=True
)
```

### Test Query 3: With History Tracking
```python
result = adv_rag.query(
    "Explain industrial design registration process",
    top_k=5, stream=True, summarize=True
)
print(f"Total queries in history: {len(result['history'])}")
```

---

## 🎓 How RAG Works

**Retrieval-Augmented Generation** combines information retrieval with text generation:

1. **Retrieval Phase**: Find the most relevant documents related to the user's query
2. **Augmentation Phase**: Use retrieved documents as context
3. **Generation Phase**: Feed context to LLM for accurate, source-backed answers

**Benefits:**
- ✅ **Reduces Hallucinations**: Answers grounded in documents
- ✅ **Up-to-date Answers**: Can reference latest documents
- ✅ **Private Data Support**: Works with proprietary documents
- ✅ **Source Attribution**: Know where answers come from
- ✅ **Cost Efficient**: Smaller LLMs can work with context

---

## 🔮 Future Enhancements

- [ ] Multi-language support (translate documents)
- [ ] Hybrid search (keyword + semantic)
- [ ] Fine-tuning with custom datasets
- [ ] GraphRAG for knowledge graph integration
- [ ] Real-time document indexing
- [ ] Streaming video/audio support
- [ ] Multi-turn conversation memory
- [ ] Custom prompt templates
- [ ] Performance monitoring & analytics
- [ ] Docker containerization

---

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact & Support

- **Author**: Akhil Shibu
- **Email**: [akhilshibu2710@gmail.com]
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Start a discussion for feature requests

---

## 🌟 Acknowledgments

Built with:
- 🦜 [LangChain](https://python.langchain.com/)
- 🤖 [Google Generative AI](https://ai.google.dev/)
- 🔍 [ChromaDB](https://www.trychroma.com/)
- 📊 [Sentence Transformers](https://www.sbert.net/)
- 🔎 [FAISS](https://github.com/facebookresearch/faiss)

---

<div align="center">

**Made with ❤️ by Akhil Shibu**

[![Stars](https://img.shields.io/github/stars/yourusername/YTRAG?style=social)](https://github.com/yourusername/YTRAG)
[![Forks](https://img.shields.io/github/forks/yourusername/YTRAG?style=social)](https://github.com/yourusername/YTRAG)

</div>
