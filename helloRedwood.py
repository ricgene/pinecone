import pinecone
import os
import sys
from dotenv import load_dotenv

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        print("You can copy .env-example to .env and fill in your values")
        sys.exit(1)

def validate_pinecone_environment(environment):
    """Validate Pinecone environment value."""
    valid_environments = [
        "gcp-starter",  # Free tier
        "us-west1-gcp", "us-east1-gcp",  # GCP regions
        "us-east-1-aws", "us-west-2-aws"  # AWS regions
    ]
    
    if environment not in valid_environments:
        print(f"Warning: Environment '{environment}' is not in the list of known valid environments.")
        print("Known valid environments are:")
        for env in valid_environments:
            print(f"- {env}")
        print("\nIf you're using a different environment, please verify it's correct.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

def main():
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    check_environment_variables()
    
    # Get environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    
    # Validate environment
    validate_pinecone_environment(environment)
    
    try:
        # Initialize Pinecone
        print(f"Initializing Pinecone with environment: {environment}")
        pinecone.init(api_key=api_key, environment=environment)
        print("Successfully connected to Pinecone")
    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if your API key is correct")
        print("2. Verify your environment value")
        print("3. Ensure you have an active internet connection")
        sys.exit(1)

    # Create an index
    index_name = "hello-world"
    dimension = 1536

    try:
        # Create index if it doesn't exist
        pinecone.create_index(index_name, dimension=dimension)
        print(f"Created index: {index_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Index {index_name} already exists, continuing...")
        else:
            print(f"Error creating index: {str(e)}")
            sys.exit(1)

    try:
        # Connect to the index
        index = pinecone.Index(index_name)
        print(f"Connected to index: {index_name}")

        # Upsert a vector
        vector = [0.1] * dimension
        index.upsert([("id1", vector)])
        print("Successfully upserted vector")

        # Query the index
        result = index.query(vector, top_k=1)
        print("\nQuery result:")
        print(result)
        
    except Exception as e:
        print(f"Error during index operations: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
