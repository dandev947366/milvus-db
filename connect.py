from pymilvus import connections, utility

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")  # or "127.0.0.1"

# Check connection by listing collections
print("Connected successfully!")
print("Collections:", utility.list_collections())
