from pymilvus import (
    connections,
    utility,
    Collection,
)
import random

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Get the collection
collection_name = "Album1"
if not utility.has_collection(collection_name):
    raise ValueError(f"Collection '{collection_name}' does not exist")

collection = Collection(collection_name)

# Load collection before search operations
collection.load()

# 1. Basic Vector Similarity Search
print("Performing basic vector search...")
try:
    # Generate a random query vector matching the dimension (64) from your schema
    query_vector = [[random.random() for _ in range(64)]]

    search_params = {
        "metric_type": "L2",  # Should match your index metric type
        "params": {"nprobe": 10},  # Number of clusters to search
    }

    results = collection.search(
        data=query_vector,
        anns_field="song_vec",
        param=search_params,
        limit=5,
        output_fields=["name", "id"],
        consistency_level="Strong",
    )

    print(f"\nBasic search results (top {len(results[0])} matches):")
    for i, hits in enumerate(results):
        print(f"\nQuery {i + 1} results:")
        for hit in hits:
            print(f"- ID: {hit.id}")
            print(f"  Name: {hit.entity.get('name')}")
            print(f"  Distance: {hit.distance:.4f}")

except Exception as e:
    print(f"Basic search failed: {e}")

# 2. Hybrid Search (Vector + Scalar Filtering)
print("\nPerforming hybrid search...")
try:
    # Generate a new query vector
    hybrid_query_vector = [[random.random() for _ in range(64)]]

    hybrid_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10, "search_k": 64},  # Fixed syntax here
    }

    hybrid_res = collection.search(
        data=hybrid_query_vector,
        anns_field="song_vec",
        param=hybrid_params,
        limit=5,
        expr="listen_count <= 100000",  # Scalar filter condition
        output_fields=["name", "id", "listen_count"],
    )

    print(f"\nHybrid search results (top {len(hybrid_res[0])} matches):")
    for i, hits in enumerate(hybrid_res):
        print(f"\nQuery {i + 1} results:")
        for hit in hits:
            print(f"- ID: {hit.id}")
            print(f"  Name: {hit.entity.get('name')}")
            print(f"  Listen Count: {hit.entity.get('listen_count')}")
            print(f"  Distance: {hit.distance:.4f}")

except Exception as e:
    print(f"Hybrid search failed: {e}")
    raise

finally:
    # Release collection
    collection.release()
    print("\nCollection released")
