from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

def initialize_components():
    """Initialize all necessary components for the RAG system."""
    try:
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        
        # Initialize Google AI components
        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME", "gemini-pro"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )
        
        return {
            "pinecone": pinecone,
            "embeddings": embeddings,
            "llm": llm,
            "text_splitter": text_splitter
        }
    
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return None

def load_pdf(file_path):
    """Load and extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        return None

def main():
    # Check if all required environment variables are set
    required_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return
    
    # Initialize components
    components = initialize_components()
    if not components:
        return
    
    print("Successfully initialized all components!")
    print("\nAvailable components:")
    print("- Pinecone client")
    print("- Google AI Embeddings")
    print("- Google AI LLM")
    print("- Text Splitter")

if __name__ == "__main__":
    main() 