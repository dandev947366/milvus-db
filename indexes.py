from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
import time

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Get the collection
collection_name = "Album1"
if not utility.has_collection(collection_name):
    raise ValueError(f"Collection '{collection_name}' does not exist")

collection = Collection(collection_name)

# Load collection before index operations
collection.load()

# 1. Vector Index Operations
if not any(index.field_name == "song_vec" for index in collection.indexes):
    print("Creating vector index...")
    vector_index_params = {
        "metric_type": "L2",
        "index_type": "IVF_SQ8",
        "params": {"nlist": 64},
    }
    collection.create_index(
        field_name="song_vec",
        index_params=vector_index_params,
        index_name="vector_index",
    )

    # Wait for index to be built
    start_time = time.time()
    while True:
        progress = utility.index_building_progress(collection_name)
        if progress["total"] == progress["indexed_rows"]:
            break
        if time.time() - start_time > 30:  # 30 second timeout
            raise TimeoutError("Vector index building timed out")
        time.sleep(0.5)
    print("Vector index created successfully")
else:
    print("Vector index already exists")

# 2. Scalar Index Operations
if not any(index.field_name == "name" for index in collection.indexes):
    print("Creating scalar index...")
    collection.create_index(
        field_name="name",
        index_name="scalar_index_name",
        index_params={"index_type": "INVERTED"},
    )
    print("Scalar index created successfully")
else:
    print("Scalar index already exists")

# Print current indexes
print("\nCurrent indexes:")
for index in collection.indexes:
    print(f"- Field: {index.field_name}")
    print(f"  Name: {index.index_name}")
    print(f"  Type: {index.params['index_type']}")
    if "metric_type" in index.params:
        print(f"  Metric: {index.params['metric_type']}")
    if "params" in index.params:
        print(f"  Params: {index.params['params']}")

# Release collection
collection.release()
