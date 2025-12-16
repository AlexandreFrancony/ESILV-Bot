# 🎯 PROJECT EXECUTIVE SUMMARY

**Project:** ESILV Smart Assistant (Multi-Agent Chatbot)  
**Date:** 16 Décembre 2025  

---

## 📌 One-Page Overview

### What We're Building
A **multi-agent AI chatbot** that helps prospective ESILV students with:
- Questions about programs, admissions, courses
- Registration & contact information collection
- Intelligent routing to appropriate agents

### Architecture: 5 Specialized Agents

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **ROUTER** | Intent classification | User message | Intent type + confidence |
| **RETRIEVER** | Knowledge base search (RAG) | Query | Top-5 relevant documents |
| **QA GENERATOR** | Answer synthesis | Query + context | Grounded, natural answer |
| **FORM-FILLER** | User info extraction | Messages | Structured JSON data |
| **CONV MANAGER** | State management | History | Updated profile + context |

### Tech Stack (Finalized)

```
Frontend:  Streamlit (web UI)
Backend:   FastAPI (optional REST API)
LLM:       Ollama + Mistral 7B (local inference, ~4.1GB)
Embedding: Ollama + Nomic-Embed-Text (274MB, 768-dim vectors)
Vector DB: Chromadb (embedded, persistent)
Orchestration: LangGraph (multi-agent routing)
Data Source: Web scraping ESILV website (BeautifulSoup)
Testing: Pytest + evaluation notebook
```

### Data Pipeline

```
ESILV Website 
    ↓ BeautifulSoup scraper
Documents (raw text)
    ↓ Semantic chunking (3-level)
Chunks (150-300 tokens, ~50-100 overlap)
    ↓ Nomic embeddings (768-dim)
Vector Store (Chromadb)
    ↓ Retrieval by cosine similarity
Top-K relevant docs → LLM for answer
```

### Why These Choices?

| Decision | Why |
|----------|-----|
| **Mistral 7B** | Best RAG quality + form-filling + speed (4.1GB VRAM) |
| **Nomic-Embed** | Lightweight (274MB), high quality, open-source |
| **Chromadb** | Embedded DB, no setup, supports metadata filtering |
| **LangGraph** | Better than chains for conditional routing |
| **Hybrid chunking** | Semantic coherence + fast retrieval + high accuracy |
| **5 agents** | Single Responsibility Principle → easy to test/debug |

### Evaluation Metrics objectives

- Retrieval Accuracy: **> 85%**
- Answer Relevance: **> 80%**
- Query Latency: **< 2s** (per query)
- Intent Classification F1: **> 90%**
- Form-Filling Accuracy: **> 95%**

### Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| **1** | ✅ Today | Planning + architecture |
| **2** | Week 1 | Data ingestion + indexing |
| **3** | Week 2 | Agent implementation + orchestration |
| **4** | Week 3 | UI (Streamlit) + deployment |
| **5** | Week 4 | Testing + report + slides |
| **Total** | ~4 weeks | **Working demo ready** |
