import pinecone
import os
from dotenv import load_dotenv

def print_environment_info():
    """Print information about Pinecone environments."""
    print("\nPinecone Environment Information:")
    print("--------------------------------")
    print("Free Tier:")
    print("- gcp-starter")
    print("\nGCP Regions:")
    print("- us-west1-gcp (US West)")
    print("- us-east1-gcp (US East)")
    print("\nAWS Regions:")
    print("- us-east-1-aws (US East)")
    print("- us-west-2-aws (US West)")
    print("\nTo find your environment:")
    print("1. Go to https://app.pinecone.io/")
    print("2. Sign in to your account")
    print("3. Look for 'Environment' in Project Settings or API Keys section")
    print("--------------------------------\n")

def test_pinecone_connection():
    """Test connection to Pinecone and list available indexes."""
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    
    if not api_key or not environment:
        print("Error: Missing Pinecone credentials in .env file")
        print("\nPlease set these variables in your .env file:")
        print("PINECONE_API_KEY=your-api-key-here")
        print("PINECONE_ENVIRONMENT=your-environment-here")
        print_environment_info()
        return
    
    try:
        # Initialize Pinecone
        print(f"Connecting to Pinecone environment: {environment}")
        pinecone.init(api_key=api_key, environment=environment)
        
        # List all indexes
        print("\nAvailable indexes:")
        indexes = pinecone.list_indexes()
        for index in indexes:
            print(f"- {index}")
            
        # Get index statistics
        if indexes:
            print("\nIndex statistics:")
            for index_name in indexes:
                index = pinecone.Index(index_name)
                stats = index.describe_index_stats()
                print(f"\nIndex: {index_name}")
                print(f"Total vectors: {stats.total_vector_count}")
                print(f"Dimension: {stats.dimension}")
                print(f"Index type: {stats.index_type}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if your API key is correct")
        print("2. Verify your environment value")
        print("3. Ensure you have an active internet connection")
        print_environment_info()

if __name__ == "__main__":
    test_pinecone_connection() 