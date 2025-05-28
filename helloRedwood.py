import pinecone

pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_ENVIRONMENT")

# Create an index
pinecone.create_index("hello-world", dimension=1536)

# Connect to the index
index = pinecone.Index("hello-world")

# Upsert a vector
index.upsert([("id1", [0.1]*1536)])

# Query the index
result = index.query([0.1]*1536, top_k=1)
print(result)
