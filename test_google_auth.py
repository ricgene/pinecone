import os
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
import json

def test_google_auth():
    """Test Google Cloud authentication and list buckets."""
    # Load environment variables
    load_dotenv()
    
    # Get credentials path
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    
    print("Debug Information:")
    print(f"Credentials path: {credentials_path}")
    print(f"Project ID: {project_id}")
    
    if not credentials_path or not project_id:
        print("Error: Missing Google Cloud credentials in .env file")
        print("\nPlease set these variables in your .env file:")
        print("GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json")
        print("GOOGLE_PROJECT_ID=your-project-id")
        return
    
    # Verify credentials file exists and is readable
    if not os.path.exists(credentials_path):
        print(f"Error: Credentials file not found at {credentials_path}")
        return
    
    try:
        # Try to read the credentials file
        with open(credentials_path, 'r') as f:
            creds_data = json.load(f)
            print("\nCredentials file contents:")
            print(f"Project ID in file: {creds_data.get('project_id')}")
            print(f"Client email: {creds_data.get('client_email')}")
        
        # Initialize Google Cloud client
        print(f"\nConnecting to Google Cloud project: {project_id}")
        client = storage.Client()
        
        # List buckets
        print("\nAvailable buckets:")
        buckets = list(client.list_buckets())
        for bucket in buckets:
            print(f"- {bucket.name}")
            
    except json.JSONDecodeError:
        print("Error: Invalid JSON in credentials file")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if your service account key file exists and is readable")
        print("2. Verify your project ID")
        print("3. Ensure the service account has necessary permissions")
        print("4. Check if you have an active internet connection")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_google_auth() 