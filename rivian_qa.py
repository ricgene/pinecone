import os
from dotenv import load_dotenv
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import tempfile
from google.cloud import storage

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download a file from GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def process_pdf_and_qa():
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Download PDF from GCS
        print("Downloading PDF from GCS...")
        download_from_gcs(
            "vso1-prizmpoc",
            "draft-ea-2251-rivian-stanton-springs-north-2024-10.pdf",
            temp_filename
        )
        
        # Load and process PDF
        print("Loading PDF...")
        loader = PyPDFLoader(temp_filename)
        pages = loader.load()
        
        # Split text into chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        
        # Initialize embeddings
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create or get Pinecone index
        index_name = "rivian-qa"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=384,  # dimension for all-MiniLM-L6-v2
                metric="cosine"
            )
        
        # Create vector store
        print("Creating vector store...")
        vectorstore = Pinecone.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name
        )
        
        # Ask a question
        print("\nAsking about water saving plans...")
        query = "What are Rivian's water saving plans or water conservation measures in their Georgia facility?"
        docs = vectorstore.similarity_search(query)
        
        print("\nRelevant information found:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(doc.page_content)
            
    finally:
        # Clean up temporary file
        os.unlink(temp_filename)

if __name__ == "__main__":
    process_pdf_and_qa() 