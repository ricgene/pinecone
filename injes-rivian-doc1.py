import os
import openai   
from google.cloud import storage
import fitz
#from pymupdf import fitz

import os
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
import json
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

def google_auth():
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
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    
    print("Successfully connected to Google Cloud")
    
google_auth()



def download_pdf_from_gcs(bucket_name, blob_name, destination_file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file)
    print(f"Downloaded {blob_name} to {destination_file}")



def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        all_text.append(text)
    return all_text

def chunk_text(texts, chunk_size=500):
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)
    return chunks

def embed_chunks(chunks, model="text-embedding-ada-002"):
    embedder = OpenAIEmbeddings()
    embeddings = embedder.embed_documents(chunks)
    return embeddings




def upsert_to_pinecone(index_name, chunks, embeddings, api_key):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    vectors = [
        {"id": f"doc-chunk-{i}", "values": emb, "metadata": {"text": chunk}}
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors)
    print(f"Upserted {len(vectors)} vectors to Pinecone.")


# Step 1: Download
# bucket = "vso1-prizmpoc"
blob = "draft-ea-2251-rivian-stanton-springs-north-2024-10.pdf"
local_pdf = "local.pdf"
bucket = "vso1-prizmpoc"
download_pdf_from_gcs(bucket, blob, local_pdf)

# Step 2: Extract text
pages = extract_text_from_pdf(local_pdf)

# Step 3: Chunk
chunks = chunk_text(pages, chunk_size=500)

# Step 4: Embed
embeddings = embed_chunks(chunks)

# Step 5: Upsert
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "vso-index-1"
upsert_to_pinecone(index_name, chunks, embeddings, pinecone_api_key)
