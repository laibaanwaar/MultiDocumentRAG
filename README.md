# Multi-Document Legal RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system for answering legal questions from PDF documents using Large Language Models (LLMs), vector search, and semantic retrieval.

The current implementation is optimized for the **Pakistan Penal Code (PPC)** and is designed to be extended into a **Multi-Document Legal Knowledge Base** supporting multiple Acts and legal documents.

---

# Features

- PDF document ingestion
- Advanced PDF text cleaning
- Legal section-aware chunking
- Cross-page section detection
- Metadata-aware retrieval
- Gemini Embeddings
- Qdrant Vector Database
- Intent-aware retrieval
- Adaptive retrieval depth
- Multi-query retrieval
- MMR (Maximum Marginal Relevance)
- Reciprocal Rank Fusion
- Scenario-aware legal retrieval
- Grounded answer generation
- Source citations
- CLI interface

---

# Project Architecture

```
                    PDF Documents
                           │
                           ▼
                document_loader.py
                           │
                           ▼
                  text_cleaner.py
                           │
                           ▼
                  text_splitter.py
                           │
                           ▼
                     ingest.py
                           │
                           ▼
                 Gemini Embeddings
                           │
                           ▼
                     Qdrant Vector DB
                           │
                           ▼
                     rag_chain.py
                           │
                           ▼
                     Gemini LLM
                           │
                           ▼
                     query_cli.py
```

---

# Project Structure

```
MultiDocumentRAG/

│
├── data/
│   └── documents/
│       └── Pakistan Penal Code.pdf
│
├── qdrant_storage/
│
├── rag/
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag_chain.py
│
├── ingest.py
├── query_cli.py
├── requirements.txt
├── .env
└── README.md
```

---

# Current Workflow

## Step 1

Load PDF documents

```
data/documents/
```

`document_loader.py`

---

## Step 2

Extract text from PDF
Clean extracted text
Remove

- headers
- footers
- amendment notes
- PDF artifacts
- broken words
- duplicate spaces
- footnotes
- invisible characters

↓

Preserve

- Chapters
- Sections
- Explanations
- Illustrations
- Enumerations

↓

`text_cleaner.py`

---

## Step 3

Create legal-aware chunks

The splitter first detects legal structure.

Priority:

Document

Chapter

Section

Explanation

Illustration

Paragraph

Sentence

Character split (fallback)
```

Each chunk stores metadata such as

```
document_name
page_number
page_start
page_end
section_number
section_title
chunk_number
chunk_length
```

↓

`text_splitter.py`

---

## Step 4

Generate embeddings

Current embedding model

```
Gemini Embedding 2
```

Embedding dimension

```
768
```

↓

`embeddings.py`

---

## Step 5

Store vectors

All embeddings are stored inside

```
Qdrant
```

Current collection

```
pakistan_penal_code
```

↓

`vector_store.py`

---

## Step 6

User Query

Example

```
A company accountant was entrusted with company funds and used the money for personal expenses.
```
Question classification
Legal concept detection
Query expansion
Adaptive retrieval
MMR
Context selection
Prompt generation
Gemini LLM
Grounded answer with citations

---

# Current Retrieval Pipeline

```
User Question
Question Classification
Concept Detection
Query Expansion
Vector Search
Metadata Filtering
Score Fusion
Adaptive Retrieval
MMR
Context Selection
Prompt Construction
Gemini
Grounded Answer
```

---

# Supported Question Types

- Section lookup
- Definition
- Punishment
- Comparison
- Scenario-based questions
- General legal questions

Example
What does Section 405 state?
What punishment is provided for theft?
Compare theft and robbery.
A person found a lost wallet and kept it.
Which provision may apply?

# Technologies Used

- Python
- LangChain
- Google Gemini
- Gemini Embedding 2
- Qdrant
- PyPDFLoader
- RecursiveCharacterTextSplitter
- dotenv

---

# Current Statistics

```
PDF Pages

161
```

```
Chunks

621
```

```
Embedding Dimension

768
```

```
Vector Database

Qdrant
```

---

# Current Limitations

Currently the system is optimized for a **single legal document**.

Although multiple PDFs can already be loaded, retrieval is not yet fully document-aware.

---

# Future Enhancements

## Multi-Document Support

The next version will support multiple legal documents including

- Pakistan Penal Code
- Code of Criminal Procedure
- Qanun-e-Shahadat
- Constitution of Pakistan
- PECA
- Anti-Terrorism Act

---

## Planned Improvements

- Document-aware retrieval
- Metadata filtering
- Cross-document reasoning
- Hybrid Search
- BM25 + Dense Retrieval
- Cross-Encoder Re-ranking
- Parent-Child Retrieval
- Context Compression
- Query Decomposition
- Multi-Hop Retrieval
- Citation Verification

---

# Running the Project

## Install

```bash
pip install -r requirements.txt
```

---

## Ingest Documents

```bash
python ingest.py
```

---

## Start the Assistant

```bash
python query_cli.py
```

---

# Example Questions

```
What does Section 405 state?

```

```
Which punishment is provided for theft?

```

```
Compare theft and robbery.

```

```
A government employee misappropriated public money.
Which provisions may apply?

```

```
A person found another person's wallet and later kept it.
Which section applies?

```

---

# Roadmap

- Advanced Metadata Filtering
- Multi-Document Retrieval
- Cross-Encoder Re-ranking
- Hybrid Search
- Legal Knowledge Graph
- Web API
- Streamlit UI
- React Frontend
- Docker Deployment
- Production-ready Qdrant Server

---

# Author

Developed as part of a Legal Retrieval-Augmented Generation (RAG) project for building an intelligent legal question-answering system over multiple legal documents.