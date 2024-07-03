from pymilvus import MilvusClient, DataType
import os
from dotenv import load_dotenv

load_dotenv()
vector_db_server = os.getenv('VECTOR_DB_STORE')
collection_name = os.getenv('COLLECTION_NAME')


client = MilvusClient(uri=f"http://{vector_db_server}:19530")

# 1. Create schema
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
)

# 2. Add fields to schema
schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=4096)
schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=4096)
schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=3072)


# 3. Prepare index parameters
index_params = client.prepare_index_params()

# 4. Add indexes
index_params.add_index(
    field_name="pk",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="source",
    index_type="TRIE"
)

index_params.add_index(
    field_name="vector", 
    index_type="AUTOINDEX",
    metric_type="L2",
    params={}
)

# 5. Create a collection
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
    consistency_level="Session"
    )

print(f"Collection: {collection_name} created")

