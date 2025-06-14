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

# Generate a proper 64-dimensional vector (matches your schema)
query_vector = [[random.random() for _ in range(64)]]

try:
    # Vector similarity search
    results = collection.search(
        data=query_vector,  # Use proper dimension vector
        anns_field="song_vec",
        param={
            "metric_type": "L2",
            "params": {"nprobe": 10},  # search_k is not valid for IVF indexes
        },
        limit=5,
        expr="listen_count < 100000",  # Example valid filter expression
        output_fields=["name"],  # Changed from 'song_name' to 'name' to match schema
    )

    print("\nSearch Results:")
    for hits in results:
        for hit in hits:
            print(
                f"ID: {hit.id}, Name: {hit.entity.get('name')}, Distance: {hit.distance:.4f}"
            )

except Exception as e:
    print(f"Search failed: {e}")
    raise

finally:
    # Release collection
    collection.release()
