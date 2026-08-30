import os
import json
import sqlite3
import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rag.db")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def fetch_all_chunks(conn):
    rows = conn.execute("SELECT source, content, embedding FROM chunks").fetchall()
    return [(source, content, json.loads(emb)) for source, content, emb in rows]


def get_top_chunks(query, top_k=3):
    config = Configuration(app_name="foundry_local_rag")
    try:
        FoundryLocalManager.initialize(config)
    except FoundryLocalException:
        pass
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.load()
    embedding_client = model.get_embedding_client()

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

    model.unload()
    return results[:top_k]


if __name__ == "__main__":
    question = "Why do divers wear hoods?"
    results = get_top_chunks(question, top_k=5)
    print(f"Question: {question}\n")
    for source, content, score in results:
        print(f"[{score:.3f}] ({source})")
        print(f"  {content}\n")