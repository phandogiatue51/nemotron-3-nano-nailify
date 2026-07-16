# sync_api_data.py

from src.vector_store import NailVectorStore


def sync_all_data():
    """Sync all Nailify API data to the local vector database."""
    print("Starting API data sync...")
    store = NailVectorStore()
    store.sync_with_api()

    count = store.collection.count()
    print(f"Sync completed. Total items: {count}")

    results = store.search("glitter nail art", top_k=3)
    print("\nSample search results for 'glitter nail art':")
    for result in results:
        print(
            f"  - {result.get('name')} "
            f"(category: {result.get('category')}, score: {result.get('relevance_score', 0):.2f})"
        )


if __name__ == "__main__":
    sync_all_data()
