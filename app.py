from foundry_local_sdk import FoundryLocalManager
from retrieval import get_top_chunks

THRESHOLD = 0.55


def build_context(results):
    parts = []
    for source, content, score in results:
        parts.append(f"[{source}] {content}")
    return "\n\n".join(parts)


def answer_query(query, top_k=1):
    top_chunks = get_top_chunks(query, top_k=top_k)

    best_score = top_chunks[0][2] if top_chunks else 0.0
    if best_score < THRESHOLD:
        print("Answer: I don't have enough information on this topic in my documents.")
        return

    context = build_context(top_chunks)

    manager = FoundryLocalManager.instance

    chat_model = manager.catalog.get_model("phi-3.5-mini")
    chat_model.download(lambda p: print(f"\rDownloading: {p:.1f}%", end="", flush=True))
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()
    chat_client.settings.max_tokens = 150
    chat_client.settings.temperature = 0.2

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

    print("Answer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

    sources = sorted(set(source for source, content, score in top_chunks))
    print(f"\n(Source: {', '.join(sources)})")

    chat_model.unload()


if __name__ == "__main__":
    print("Diving Knowledge Assistant - type 'q' and press Enter to quit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("See you next time!")
            break
        if not question:
            continue
        answer_query(question)
        print("\nAny other questions? (type 'q' to quit)\n")