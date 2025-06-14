from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Define fields
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

# Define schema and collection
collection_schema = CollectionSchema(
    fields=[song_name, song_id, listen_count, song_vec, song_json, song_array],
    description="Album songs",
)

if utility.has_collection("Album1"):
    utility.drop_collection("Album1")

collection = Collection(name="Album1", schema=collection_schema)

print("Collections:", utility.list_collections())

# # Rename and drop for demo
# utility.rename_collection("Album1", "Album2")
# print("Collections after rename:", utility.list_collections())

# utility.drop_collection("Album2")
# print("Collections after drop:", utility.list_collections())
