from foundry_local_sdk import Configuration, FoundryLocalManager


def test_setup():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download(lambda p: print(f"\rDownloading: {p:.1f}%", end="", flush=True))
    print()
    model.load()
    print("Model loaded successfully! Setup is working.")

    embedding_client = model.get_embedding_client()
    test_sentence = "Foundry Local runs models on-device without needing the internet."
    response = embedding_client.generate_embedding(test_sentence)
    vector = response.data[0].embedding

    print(f"Test sentence embedded. Vector size: {len(vector)}")
    model.unload()


if __name__ == "__main__":
    test_setup()