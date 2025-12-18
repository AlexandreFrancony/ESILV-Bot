# ESILV-Bot
ESILV Smart Assistant – Multi-Agent Chatbot

Deliverables:
1) **Report**: up to 10 pages, describes context, architecture, design choices, data processing,
LLM provider selection, evaluation, and future improvements.
2) **Source Code** : Full implementation with version control, clear
README. Should include: app/, agents/, ui/, ingestion/, and notebooks/.
3) **Working Demo** (Streamlit): Functional chatbot able to answer ESILV-related questions
and collect user information. A recorded demo video or live presentation required.
4) **Evaluation Notebook**: Notebook demonstrating query examples, accuracy metrics, and
latency plots.
5) **Slide Deck** (10-15 slides)

# Fresh start
To start fresh, delete the existing virtual environment folder (e.g., `venv/` or `env/`), then create a new one using:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

After this, you'll need to build your data throught the scrapper and chunker again.
```bash
python src/ingestion/pdf_loader.py
python src/ingestion/chunker.py
python src/ingestion/indexer.py # answer n when asked
```