import os
import openai   
from google.cloud import storage
from pymupdf import fitz

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
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(input=chunk, model=model)
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

from pinecone import Pinecone

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
