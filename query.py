import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ── Config ─────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "anthropic-docs"
EMBED_MODEL      = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"
TOP_K            = 5

# ── Clients ────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
index         = pc.Index(INDEX_NAME)

SYSTEM_PROMPT = """You are an AdTech knowledge assistant. You help marketers, 
advertisers, and publishers understand programmatic advertising, digital marketing 
concepts, privacy regulations, and ad technology.

Answer questions using ONLY the context provided below. If the answer is not 
in the context, say "I don't have information on that in my knowledge base."

Always be specific and cite which topic your answer comes from where relevant.
Keep answers clear and concise — you are explaining AdTech concepts to marketing 
professionals, not engineers."""


def embed(text):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def query(question):
    # Embed the question
    vector = embed(question)

    # Retrieve top K chunks from Pinecone
    results = index.query(
        vector=vector,
        top_k=TOP_K,
        include_metadata=True
    )

    # Build context from retrieved chunks
    context_parts = []
    sources = []
    for match in results.matches:
        meta = match.metadata
        context_parts.append(f"[{meta['title']}]\n{meta['text']}")
        if meta['url'] not in sources:
            sources.append(meta['url'])

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content

    print(f"\nQ: {question}")
    print(f"\nA: {answer}")
    print(f"\nSources:")
    for s in sources:
        print(f"  - {s}")
    print()


if __name__ == "__main__":
    print("=== AdTech Knowledge Assistant (CLI) ===")
    print("Type your question or 'quit' to exit\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            query(q)