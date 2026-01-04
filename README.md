# ESILV-Bot  
# ESILV Smart Assistant – Multi-Agent RAG Chatbot

## 1. Overview

ESILV-Bot is an intelligent conversational assistant for ESILV engineering school, built on a multi-agent architecture with Retrieval-Augmented Generation (RAG).

**Key Features:**
- Answers questions about programs, admissions, and courses (RAG)
- Intelligent contact information collection
- Automatic user intent classification
- Intuitive Streamlit chat interface
- Admin dashboard for contact management

## 2. Architecture

### 2.1 Implemented Agents

| Agent | Role | Technologies |
|-------|------|--------------|
| **RouterAgent** | Intent classification (7 types) | Mistral 7B via Ollama |
| **RetrieverAgent** | Semantic search (RAG) | ChromaDB + Nomic embeddings |
| **QAGeneratorAgent** | Answer generation | Mistral 7B via Ollama |
| **FormFillerAgent** | Conversational data collection | Mistral 7B + Regex validation |

### 2.2 Processing Workflow

```
User Query
    ↓
RouterAgent (intent classification)
    ↓
    ├─→ [program_info, admission_help, course_details, general_info]
    │       ↓
    │   RetrieverAgent (search ChromaDB)
    │       ↓
    │   QAGeneratorAgent (generate answer)
    │
    ├─→ [small_talk]
    │       ↓
    │   QAGeneratorAgent (direct response)
    │
    └─→ [contact_collection]
            ↓
        FormFillerAgent (multi-turn dialogue)
            ↓
        Storage (data/contacts.json)
```

### 2.3 Supported Intents

1. **program_info**: Questions about programs/courses
2. **admission_help**: Questions about admissions/applications
3. **course_details**: Questions about courses/modules
4. **contact_collection**: Contact/brochure requests
5. **general_info**: General information about ESILV
6. **small_talk**: Greetings/politeness
7. **unknown**: Unclassifiable intent

## 3. Project Structure

```
ESILV-Bot/
├── data/
│   ├── esilv_documents.txt      # Source documents (extracted PDFs)
│   ├── chunks.json              # Indexed chunks
│   ├── chromadb/                # Vector store
│   └── contacts.json            # Collected contacts
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py        # Multi-file PDF extraction
│   │   ├── chunker.py           # Semantic chunking
│   │   └── indexer.py           # ChromaDB indexing
│   ├── agents/
│   │   ├── router.py            # Intent classification
│   │   ├── retriever.py         # RAG search
│   │   ├── qa_generator.py      # Answer generation
│   │   └── form_filler.py       # Contact collection
│   ├── core/
│   │   └── workflow.py          # Orchestration (WorkflowManager)
│   ├── ui/
│   │   ├── app.py               # Main Streamlit interface
│   │   └── admin.py             # Admin dashboard (contacts)
│   └── utils/
│       └── save_data.py         # Contact storage
├── logs/                         # Agent logs
├── notebooks/                    # Evaluation and tests
├── requirements.txt
└── README.md
```

## 4. Installation

### 4.1 Prerequisites

- Python >= 3.9
- Ollama installed and running
- ~6GB disk space (models + data)

### 4.2 Dependency Installation

```bash
# Clone repository
git clone https://github.com/AlexandreFrancony/ESILV-Bot.git
cd ESILV-Bot

# Create virtual environment
python -m venv venv

# Activate environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Ollama Installation and Configuration

```bash
# Download Ollama: https://ollama.com/download

# Start Ollama server
ollama serve

# In another terminal, download models
ollama pull mistral:7b-instruct-q4_0
ollama pull nomic-embed-text

# Verify installation
ollama list
```

## 5. Data Ingestion Pipeline

ESILV PDFs must be placed in the `data/` folder with these names:
- `esilv_brochure.pdf`
- `esilv-msc.pdf`
- `bachelors-esilv.pdf`
- `plaquette_apprentissage_esilv.pdf`
- `livret_pedagogique_esilv.pdf`

**To rebuild the index (after adding/modifying PDFs):**

```bash
# 1. Extract text from PDFs
python src/ingestion/pdf_loader.py

# 2. Split into semantic chunks
python src/ingestion/chunker.py

# 3. Index into ChromaDB
python src/ingestion/indexer.py
```

## 6. Usage

### 6.1 Streamlit Interface (Main Chat)

```bash
streamlit run src/ui/app.py
```

**Features:**
- Chat with the assistant
- Display detected intent
- Cited sources with similarity scores
- Response time metrics
- Contact widget (toggle top right)

### 6.2 Admin Dashboard (Contact Management)

```bash
streamlit run src/ui/admin.py
```

**Features:**
- Collected contact statistics
- Filter by request type and date
- CSV/Excel export
- Email list for bulk sending
- Delete all contacts

### 6.3 CLI Mode (Workflow Testing)

```bash
python -m src.core.workflow
```

Command-line interface for quick agent testing.

## 7. FormFillerAgent - Contact Collection

### 7.1 How It Works

The FormFillerAgent intelligently collects contact information through multi-turn dialogue:

```
User: "I'd like to receive brochures"
Bot: "With pleasure! I need some information. What's your first name?"

User: "Pierre"
Bot: "And your last name?"

User: "Dupont"
Bot: "What's your email address?"

User: "pierre.dupont@gmail.com"
Bot: "What's your phone number?"

User: "06 12 34 56 78"
Bot: "Thank you Pierre! Brochures will be sent to pierre.dupont@gmail.com."
```

### 7.2 Supported Request Types

- **brochures**: Documentation sending
- **callback**: Callback request
- **visit**: Campus visit
- **meeting**: Meeting with advisor
- **application**: Application

### 7.3 Data Validation

- **Email**: Regex validation (standard format)
- **Phone**: French format (10 digits, starts with 0)
- **Clear error messages** for invalid format

### 7.4 Storage

Contacts are saved in `data/contacts.json`:

```json
[
  {
    "first_name": "Pierre",
    "last_name": "Dupont",
    "email": "pierre.dupont@gmail.com",
    "phone": "0612345678",
    "request_type": "brochures",
    "timestamp": "2026-01-04T14:30:22"
  }
]
```

## 8. Configuration and Parameters

### 8.1 LLM Models

| Model | Usage | Size |
|--------|-------|--------|
| **mistral:7b-instruct-q4_0** | Classification, QA, FormFiller | ~4.1 GB |
| **nomic-embed-text** | Vector embeddings | ~274 MB |

### 8.2 RAG Parameters

```python
# src/agents/retriever.py
DEFAULT_TOP_K = 5           # Number of retrieved documents

# src/agents/qa_generator.py
TEMPERATURE = 0.3           # Answer determinism
MAX_TOKENS = 500            # Max answer length

# src/ingestion/chunker.py
CHUNK_SIZE = 250            # Tokens per chunk
MIN_CHUNK_SIZE = 50         # Minimum chunk size
```

## 9. Performance

### 9.1 Typical Response Times

| Operation | Time | Notes |
|-----------|-------|-------|
| Initial loading | ~10s | Once at startup |
| Intent classification | ~1s | RouterAgent |
| Complete RAG query | ~8-10s | Retrieval + generation |
| Small talk | ~3-5s | No retrieval |
| FormFiller (per field) | ~5-8s | Extraction + validation |

### 9.2 Implemented Optimizations

- **Streamlit cache** (`@st.cache_resource`) for agents
- **Agent preloading** at startup
- **Lazy loading** of retriever for small_talk
- **Progressive display** (intent then answer)

## 10. Evaluation

### 10.1 Target Metrics

- **Retrieval Accuracy**: > 85%
- **Answer Relevance**: > 80%
- **Query Latency**: < 10s
- **Intent Classification F1**: > 90%
- **FormFilling Accuracy**: > 95%

### 10.2 Evaluation Notebooks

```bash
# Open Jupyter notebooks
jupyter notebook notebooks/
```

- `evaluations.ipynb`: Performance metrics
- `retrieval_testing.ipynb`: Retrieval tests
- `agent_testing.ipynb`: Agent tests

## 11. Logs and Debugging

All logs are stored in `logs/`:

```
logs/
├── workflow_20260104_123045.log      # Orchestration
├── router_20260104_123045.log        # Classification
├── retriever_20260104_123045.log     # RAG
├── qa_generator_20260104_123045.log  # Generation
└── ...
```

**Log levels:**
- **DEBUG**: Technical details
- **INFO**: Main operations
- **WARNING**: Unexpected behaviors
- **ERROR**: Blocking errors

## 12. Troubleshooting

### 12.1 Ollama not responding

```bash
# Check if Ollama is running
ollama ps

# Restart if necessary
ollama serve
```

### 12.2 Missing models

```bash
# List installed models
ollama list

# Download missing models
ollama pull mistral:7b-instruct-q4_0
ollama pull nomic-embed-text
```

### 12.3 Empty ChromaDB

```bash
# Rebuild index
python src/ingestion/indexer.py
```

### 12.4 Streamlit can't find agents

```bash
# Stop Streamlit (Ctrl+C)
# Clear cache
rm -rf ~/.streamlit/cache
# Restart
streamlit run src/ui/app.py
```

## 13. Technologies Used

| Category | Technology | Version |
|-----------|-------------|---------|
| **LLM** | Ollama + Mistral 7B | latest |
| **Embeddings** | Nomic-embed-text | latest |
| **Vector DB** | ChromaDB | >=0.4.0 |
| **Frontend** | Streamlit | >=1.28.0 |
| **PDF Processing** | PyPDF2, pdfplumber | latest |
| **Language** | Python | >= 3.9 |