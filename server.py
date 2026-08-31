import os
import json
import sqlite3
import math
from flask import Flask, request, jsonify, send_from_directory
from foundry_local_sdk import Configuration, FoundryLocalManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rag.db")
THRESHOLD = 0.55

app = Flask(__name__, static_folder="static")

config = Configuration(app_name="foundry_local_rag")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("Loading embedding model...")
embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
embedding_model.load()
embedding_client = embedding_model.get_embedding_client()

print("Loading chat model...")
chat_model = manager.catalog.get_model("phi-3.5-mini")
chat_model.load()
chat_client = chat_model.get_chat_client()
chat_client.settings.max_tokens = 150
chat_client.settings.temperature = 0.2

print("Models ready. Starting server.")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def fetch_all_chunks(conn):
    rows = conn.execute("SELECT source, content, embedding FROM chunks").fetchall()
    return [(source, content, json.loads(emb)) for source, content, emb in rows]


def get_top_chunks(query, top_k=1):
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    conn = sqlite3.connect(DB_PATH)
    chunks = fetch_all_chunks(conn)
    conn.close()

    results = []
    for source, content, emb in chunks:
        score = cosine_similarity(query_embedding, emb)
        results.append((source, content, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


def build_context(results):
    parts = []
    for source, content, score in results:
        parts.append(f"[{source}] {content}")
    return "\n\n".join(parts)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("question") or "").strip()

    if not query:
        return jsonify({"answer": "Please type a question first.", "sources": []})

    top_chunks = get_top_chunks(query, top_k=1)
    best_score = top_chunks[0][2] if top_chunks else 0.0

    if best_score < THRESHOLD:
        return jsonify({
            "answer": "I don't have enough information on this topic in my documents.",
            "sources": [],
        })

    context = build_context(top_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the given context. "
                "Write 1-2 complete, clear sentences. "
                "If the context does not contain enough information, say you don't know; do not make up an answer.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    answer_text = ""
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            answer_text += content

    sources = sorted(set(source for source, content, score in top_chunks))
    return jsonify({"answer": answer_text.strip(), "sources": sources})


if __name__ == "__main__":
    app.run(debug=False, port=5000)