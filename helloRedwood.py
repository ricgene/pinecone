import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # Loads variables from .env

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ["PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        sys.exit(1)

def main():
    check_environment_variables()
    api_key = os.getenv("PINECONE_API_KEY")
    print("API Key:", api_key)  # Should not be None

    # Set cloud and region for serverless index
    cloud = "aws"  # or "gcp"
    region = "us-east-1"  # e.g., "us-east-1" for AWS, "us-east1" for GCP

    pc = Pinecone(api_key=api_key)
    index_name = "hello-world"
    dimension = 1536
    metric = "cosine"

    try:
        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"Created index: {index_name}")
        else:
            print(f"Index {index_name} already exists, continuing...")
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        sys.exit(1)

    try:
        # Connect to the index
        index = pc.Index(index_name)
        print(f"Connected to index: {index_name}")

        # Upsert a vector
        vector = [0.1] * dimension
        index.upsert([{"id": "id1", "values": vector}])
        print("Successfully upserted vector")

        # Query the index
        result = index.query(
            vector=vector,
            top_k=1,
            include_values=True
        )
        print("\nQuery result:")
        print(result)
        
    except Exception as e:
        print(f"Error during index operations: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
