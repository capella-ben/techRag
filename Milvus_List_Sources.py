from pymilvus import connections, Collection
import os
from dotenv import load_dotenv

load_dotenv()
vector_db_server = os.getenv('VECTOR_DB_STORE')
collection_name = os.getenv('COLLECTION_NAME')

# Connect to the Milvus server
connections.connect(
    alias="default",
    uri=f"http://{vector_db_server}:19530"
)

# Load the collection
collection = Collection(name=collection_name)

# Query to get distinct values in the "source" field
results = collection.query(
    expr='source != "Fact"',
    output_fields=["source"],
    distinct=True, limit=5000
)

sources = list({item["source"] for item in results})
#print("Distinct values in the 'source' field:")
sources.sort()
for s in sources:
    print(s)

print()
print("Count of distinct sources:", len(sources))


