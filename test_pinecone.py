from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # Make sure this is before os.environ.get

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Your Pinecone code continues here...
