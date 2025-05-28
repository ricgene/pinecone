# RAG System with Pinecone and Google AI

A Retrieval-Augmented Generation (RAG) system built with LangChain, Pinecone, and Google's Generative AI. This system allows you to create a knowledge base from your documents and query it using natural language.

## Features

- Document processing with PDF support
- Vector storage using Pinecone
- Text embeddings with Google AI
- Natural language querying with Gemini Pro
- Configurable chunk sizes and overlap
- Environment-based configuration

## Prerequisites

- Python 3.11 or higher
- Pinecone API key (get from [Pinecone Console](https://app.pinecone.io/))
- Google AI API key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env-example .env
```
Edit `.env` with your API keys and preferred settings.

## Project Structure

```
.
├── .env-example          # Example environment variables
├── .gitignore           # Git ignore rules
├── requirements.txt     # Project dependencies
├── rag_setup.py        # Main RAG system setup
└── README.md           # This file
```

## Usage

1. Initialize the RAG system:
```python
from rag_setup import initialize_components

# Initialize components
components = initialize_components()
if components:
    pinecone = components["pinecone"]
    embeddings = components["embeddings"]
    llm = components["llm"]
    text_splitter = components["text_splitter"]
```

2. Process a PDF document:
```python
from rag_setup import load_pdf

# Load and process PDF
text = load_pdf("path/to/your/document.pdf")
if text:
    # Split text into chunks
    chunks = text_splitter.split_text(text)
```

## Configuration

The system can be configured through environment variables in `.env`:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `PINECONE_INDEX_NAME`: Name for your vector index
- `GOOGLE_API_KEY`: Your Google AI API key
- `MODEL_NAME`: LLM model name (default: gemini-pro)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-004)
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TEMPERATURE`: LLM response temperature (default: 0.7)
- `MAX_OUTPUT_TOKENS`: Maximum response tokens (default: 2048)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Google AI](https://ai.google.dev/) 