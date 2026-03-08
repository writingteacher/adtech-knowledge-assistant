import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
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

# ── HTML template ──────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AdTech Knowledge Assistant</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f0f2f5;
      height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }

    .container {
      width: 100%;
      max-width: 720px;
      height: 90vh;
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .header {
      background: #1a1a2e;
      color: white;
      padding: 20px 24px;
      flex-shrink: 0;
    }

    .header h1 {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .header p {
      font-size: 13px;
      color: #8888aa;
    }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.6;
    }

    .message.user {
      background: #1a1a2e;
      color: white;
      align-self: flex-end;
      border-bottom-right-radius: 4px;
    }

    .message.assistant {
      background: #f5f5f0;
      color: #1a1a2e;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
    }

    .sources {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #e0e0d8;
      font-size: 12px;
      color: #666688;
    }

    .sources a {
      color: #3a82b8;
      text-decoration: none;
      display: block;
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sources a:hover { text-decoration: underline; }

    .input-area {
      padding: 16px 24px;
      border-top: 1px solid #eee;
      display: flex;
      gap: 10px;
      flex-shrink: 0;
    }

    .input-area input {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid #ddd;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
    }

    .input-area input:focus { border-color: #3a82b8; }

    .input-area button {
      padding: 12px 20px;
      background: #1a1a2e;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
      font-weight: 600;
    }

    .input-area button:hover { background: #2a2a4e; }
    .input-area button:disabled { background: #aaa; cursor: not-allowed; }

    .welcome {
      text-align: center;
      color: #9999aa;
      font-size: 13px;
      margin: auto;
      padding: 40px 20px;
    }

    .welcome h2 { font-size: 16px; color: #555; margin-bottom: 8px; }

    .suggestions {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 16px;
    }

    .suggestion {
      background: #f5f5f0;
      border: 1px solid #e0e0d8;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      color: #3a82b8;
      cursor: pointer;
      text-align: left;
    }

    .suggestion:hover { background: #eeeee8; }

    .typing {
      font-style: italic;
      color: #9999aa;
      font-size: 13px;
    }

    .footer {
      text-align: center;
      font-size: 11px;
      color: #bbbbcc;
      padding: 8px;
      flex-shrink: 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>AdTech Knowledge Assistant</h1>
      <p>Ask questions about programmatic advertising, privacy regulations, and ad technology</p>
    </div>

    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <h2>What would you like to know?</h2>
        <p>Ask anything about programmatic advertising, DSPs, SSPs, GDPR, ad fraud, and more.</p>
        <div class="suggestions">
          <div class="suggestion" onclick="askSuggestion(this)">What is the difference between a DSP and an SSP?</div>
          <div class="suggestion" onclick="askSuggestion(this)">How does GDPR affect programmatic advertising?</div>
          <div class="suggestion" onclick="askSuggestion(this)">What is header bidding and how does it work?</div>
          <div class="suggestion" onclick="askSuggestion(this)">What is ad fraud and how can it be prevented?</div>
        </div>
      </div>
    </div>

    <div class="input-area">
      <input type="text" id="input" placeholder="Ask a question about AdTech..." onkeydown="if(event.key==='Enter') sendMessage()">
      <button id="send-btn" onclick="sendMessage()">Ask</button>
    </div>

    <div class="footer">Built by <a href="https://rwhyte.com" style="color:#bbbbcc;">rwhyte.com</a> · RAG pipeline on Wikipedia AdTech corpus · OpenAI + Pinecone</div>
  </div>

  <script>
    function askSuggestion(el) {
      document.getElementById('input').value = el.textContent;
      sendMessage();
    }

    function addMessage(text, role, sources) {
      const welcome = document.getElementById('welcome');
      if (welcome) welcome.remove();

      const msgs = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = `message ${role}`;
      div.textContent = text;

      if (sources && sources.length > 0) {
        const src = document.createElement('div');
        src.className = 'sources';
        src.textContent = 'Sources: ';
        sources.forEach(url => {
          const a = document.createElement('a');
          a.href = url;
          a.target = '_blank';
          a.textContent = url;
          src.appendChild(a);
        });
        div.appendChild(src);
      }

      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
      return div;
    }

    async function sendMessage() {
      const input = document.getElementById('input');
      const btn   = document.getElementById('send-btn');
      const q     = input.value.trim();
      if (!q) return;

      input.value = '';
      btn.disabled = true;

      addMessage(q, 'user');

      const typing = addMessage('Thinking...', 'assistant');
      typing.classList.add('typing');

      try {
        const res  = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        typing.remove();
        addMessage(data.answer, 'assistant', data.sources);
      } catch(e) {
        typing.textContent = 'Something went wrong. Please try again.';
        typing.classList.remove('typing');
      }

      btn.disabled = false;
      input.focus();
    }
  </script>
</body>
</html>
"""

app = Flask(__name__)


def embed(text):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a question.", "sources": []})

    # Embed and retrieve
    vector  = embed(question)
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)

    context_parts = []
    sources       = []
    for match in results.matches:
        meta = match.metadata
        context_parts.append(f"[{meta['title']}]\n{meta['text']}")
        if meta["url"] not in sources:
            sources.append(meta["url"])

    context = "\n\n---\n\n".join(context_parts)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)