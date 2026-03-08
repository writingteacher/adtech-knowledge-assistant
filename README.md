# AdTech Knowledge Assistant

A RAG-based AI knowledge assistant for programmatic advertising, AdTech concepts, and digital marketing privacy regulations. Ask plain-language questions and receive accurate, sourced answers drawn exclusively from a curated Wikipedia corpus.

**Live demo:** [adtech-knowledge-assistant.onrender.com](https://adtech-knowledge-assistant.onrender.com)

> **Portfolio note:** This project was built as a portfolio piece demonstrating end-to-end RAG system design using web-crawled content. Built to production standards to illustrate scalability and knowledge architecture decisions.

---

## What it does

- Crawls 24 Wikipedia articles covering the AdTech ecosystem via the Wikipedia API
- Chunks content into 800-word segments with 150-word overlap to preserve context
- Embeds each chunk using OpenAI's `text-embedding-3-small` model
- Stores 120 vectors in a Pinecone serverless index
- Accepts natural language questions, retrieves semantically relevant chunks, and generates grounded answers via GPT-4o-mini
- Serves a chat interface via Flask, deployed on Render

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Content source | Wikipedia API |
| Embedding model | OpenAI text-embedding-3-small |
| Vector database | Pinecone (serverless) |
| LLM | OpenAI GPT-4o-mini |
| Web framework | Flask |
| Deployment | Render |

---

## Knowledge base — topic coverage

**Core programmatic**
- Online advertising, Programmatic advertising
- Demand-side platform, Supply-side platform
- Real-time bidding, Ad exchange, Header bidding, Ad serving

**Data & audiences**
- Data management platform, Customer data platform
- Lookalike audience, Retargeting, Behavioral targeting, Contextual advertising

**Metrics & measurement**
- Cost per mille, Click-through rate, Conversion tracking
- Attribution (marketing), Viewability

**Privacy & regulation**
- General Data Protection Regulation (GDPR)
- California Consumer Privacy Act (CCPA)
- Children's Online Privacy Protection Act (COPPA)
- Internet privacy, HTTP cookie, Third-party cookie
- Consent management platform

**Brand safety & fraud**
- Brand safety, Ad fraud, Ad blocking

---

## Project structure

```
adtech-knowledge-assistant/
├── crawl.py          # Fetches Wikipedia articles via API, saves to JSON
├── index_all.py      # Chunks, embeds, and upserts vectors to Pinecone
├── query.py          # Command-line query interface for testing
├── app.py            # Flask web app with chat UI
├── requirements.txt  # Python dependencies
└── .env              # API keys (not committed)
```

---

## Setup

### 1. Clone the repo

```
git clone https://github.com/writingteacher/adtech-knowledge-assistant.git
cd adtech-knowledge-assistant
```

### 2. Create and activate a virtual environment

```
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Add API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
```

### 5. Create Pinecone index

In your Pinecone dashboard, create an index with:

- **Name:** `anthropic-docs`
- **Dimensions:** `1536`
- **Metric:** `cosine`
- **Type:** Serverless, AWS us-east-1

### 6. Crawl Wikipedia articles

```
python crawl.py
```

Fetches 24 AdTech articles from the Wikipedia API and saves to `crawled_docs.json`.

### 7. Index the corpus

```
python index_all.py
```

Chunks, embeds, and upserts 120 vectors to Pinecone. Embedding cost: under $0.01.

### 8. Test retrieval

```
python query.py
```

### 9. Run the web app

```
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Deployment

Deployed on [Render](https://render.com):

1. Push code to GitHub
2. Connect repo in Render as a Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Add `OPENAI_API_KEY` and `PINECONE_API_KEY` as environment variables

---

## Key design decisions

**Why Wikipedia as a source?**
Wikipedia's content is Creative Commons licensed — no ToS issues for a portfolio project. The AdTech article cluster provides a well-structured, authoritative corpus covering the full programmatic ecosystem.

**Why 800-word chunks?**
Larger than a standard 500-word chunk to preserve contextual integrity across technical definitions and multi-part explanations. The 150-word overlap ensures content near chunk boundaries is captured in at least one retrievable segment.

**Why cosine similarity?**
Measures semantic angle between vectors rather than magnitude, making retrieval robust to differences in chunk length — important when mixing short definition articles with long regulatory articles like GDPR.

**Why GPT-4o-mini?**
Cost-efficient for high query volumes while maintaining strong instruction-following for the grounding rules in the system prompt.

**Hallucination controls**
The system prompt enforces two strict behaviours: answer only from retrieved context, and explicitly acknowledge when information is not in the knowledge base rather than guessing.

---

## Author

**Rob Whyte**
Technical Writer & AI Content Architect
[rwhyte.com](https://rwhyte.com) | [linkedin.com/in/robwhyte](https://linkedin.com/in/robwhyte/)