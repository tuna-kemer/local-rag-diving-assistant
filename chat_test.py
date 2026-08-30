from foundry_local_sdk import Configuration, FoundryLocalManager


def chat_test():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.download(lambda p: print(f"\rDownloading: {p:.1f}%", end="", flush=True))
    print()
    chat_model.load()
    print("Chat model loaded.")

    chat_client = chat_model.get_chat_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant who gives short, clear answers."},
        {"role": "user", "content": "Hello, who are you?"},
    ]

    print("Answer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()

    chat_model.unload()


if __name__ == "__main__":
    chat_test()