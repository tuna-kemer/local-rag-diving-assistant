import os
import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "belgeler")
DB_PATH = os.path.join(BASE_DIR, "rag.db")
BATCH_SIZE = 10


def read_documents():
    documents = []
    for filename in sorted(os.listdir(DOCS_FOLDER)):
        if filename.endswith(".txt"):
            path = os.path.join(DOCS_FOLDER, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append((filename, content))
    return documents


def split_into_chunks(content):
    parts = content.split("\n\n")
    parts = [p.strip() for p in parts if len(p.strip()) > 30]
    return parts


def create_chunks_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()


def prepare_all_chunks():
    documents = read_documents()
    all_chunks = []
    for name, content in documents:
        for part in split_into_chunks(content):
            all_chunks.append((name, part))
    return all_chunks


def embed_in_batches(embedding_client, texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = embedding_client.generate_embeddings(batch)
        embeddings.extend(item.embedding for item in response.data)
        done = min(i + batch_size, len(texts))
        print(f"\rEmbedded {done}/{len(texts)}", end="", flush=True)
    print()
    return embeddings


def run_ingest():
    all_chunks = prepare_all_chunks()
    print(f"Prepared {len(all_chunks)} chunks total.")

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download(lambda p: print(f"\rDownloading: {p:.1f}%", end="", flush=True))
    print()
    model.load()
    print("Embedding model loaded.")

    embedding_client = model.get_embedding_client()

    chunk_texts = [text for source, text in all_chunks]
    embeddings = embed_in_batches(embedding_client, chunk_texts)
    print(f"{len(embeddings)} chunks embedded.")

    conn = sqlite3.connect(DB_PATH)
    create_chunks_table(conn)
    conn.execute("DELETE FROM chunks")

    for i, ((source, text), emb) in enumerate(zip(all_chunks, embeddings)):
        conn.execute(
            "INSERT INTO chunks (id, source, content, embedding) VALUES (?, ?, ?, ?)",
            (i, source, text, json.dumps(emb))
        )
    conn.commit()
    print("All chunks saved to the database.")

    model.unload()
    conn.close()


if __name__ == "__main__":
    run_ingest()