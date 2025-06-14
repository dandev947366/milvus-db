from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
import string
import random
import json

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Define schema
song_name = FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200)
song_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
listen_count = FieldSchema(name="listen_count", dtype=DataType.INT64)
song_vec = FieldSchema(name="song_vec", dtype=DataType.FLOAT_VECTOR, dim=64)
song_json = FieldSchema(name="song_json", dtype=DataType.JSON)
song_array = FieldSchema(
    name="song_array",
    dtype=DataType.ARRAY,
    element_type=DataType.INT64,
    max_capacity=900,
)

collection_schema = CollectionSchema(
    fields=[song_name, song_id, listen_count, song_vec, song_json, song_array],
    description="Album songs",
)

# Create or get collection
collection_name = "Album1"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=collection_schema)
    print(f"Created new collection: {collection_name}")
else:
    collection = Collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")

# Load collection
collection.load()

# Generate complete test data matching all schema fields
num_entities = 5
data = [
    [
        "".join(random.choices(string.ascii_letters, k=7)) for _ in range(num_entities)
    ],  # song_name
    [i for i in range(num_entities)],  # song_id
    [random.randint(0, 100000) for _ in range(num_entities)],  # listen_count
    [
        [random.random() for _ in range(64)] for _ in range(num_entities)
    ],  # song_vec (64-dim)
    [
        json.dumps(
            {
                "artist": random.choice(["Artist1", "Artist2", "Artist3"]),
                "year": random.randint(1990, 2023),
            }
        )
        for _ in range(num_entities)
    ],  # song_json
    [
        [random.randint(1, 100) for _ in range(random.randint(1, 10))]
        for _ in range(num_entities)
    ],  # song_array
]

# Insert data
try:
    insert_result = collection.insert(data)
    collection.flush()  # Ensure data is persisted
    print(f"Successfully inserted {len(insert_result.primary_keys)} entities")
    print(f"Inserted IDs: {insert_result.primary_keys}")
except Exception as e:
    print(f"Insert failed: {e}")
    raise

# Delete operation (using IDs that were just inserted)
try:
    expr = "id in [0, 1]"  # Note: using 'id' (primary key field name) not 'song_id'
    delete_result = collection.delete(expr)
    print(f"Deleted {delete_result.delete_count} entities")
except Exception as e:
    print(f"Delete failed: {e}")
    raise

# Verify deletion
try:
    remaining = collection.query(expr="id in [0, 1]", output_fields=["id"])
    print(f"Entities remaining after deletion: {len(remaining)}")
except Exception as e:
    print(f"Query failed: {e}")

collection.compact()
# Release collection
collection.release()
