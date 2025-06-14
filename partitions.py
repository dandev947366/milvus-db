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

# Create or load collection
collection_name = "Album1"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=collection_schema)
    print(f"Collection '{collection_name}' created.")
else:
    collection = Collection(name=collection_name)
    print(f"Collection '{collection_name}' loaded.")

# Index creation with waiting
if not collection.has_index():
    print("Creating index on 'song_vec' field...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="song_vec", index_params=index_params)

    # Wait for index to be built
    start_time = time.time()
    timeout = 30  # seconds
    while (
        not utility.index_building_progress(collection_name)["total"]
        == utility.index_building_progress(collection_name)["indexed_rows"]
    ):
        if time.time() - start_time > timeout:
            raise TimeoutError("Index building timed out")
        time.sleep(0.5)
    print("Index created and built.")

# Load collection for partition operations
print("\nLoading collection for partition operations...")
collection.load()

# Partition operations
print("\nManaging partitions:")
partitions = ["Disc1", "Disc2"]

for partition_name in partitions:
    if not collection.has_partition(partition_name):
        print(f"Creating partition '{partition_name}'...")
        collection.create_partition(partition_name)
    else:
        print(f"Partition '{partition_name}' already exists.")

# List all partitions
print("\nCurrent partitions:")
for partition in collection.partitions:
    print(f"- {partition.name}")

# Release collection before dropping partition
print("\nReleasing collection before partition drop...")
collection.release()

# Drop Disc2 partition
print("\nDropping partition 'Disc2'...")
if collection.has_partition("Disc2"):
    collection.drop_partition("Disc2")
    print("'Disc2' partition dropped.")
else:
    print("'Disc2' partition does not exist.")

# Reload collection to verify changes
print("\nReloading collection to verify changes...")
collection.load()

# Final partition list
print("\nRemaining partitions:")
for partition in collection.partitions:
    print(f"- {partition.name}")

# Release collection
collection.release()
print("\nOperation completed.")
