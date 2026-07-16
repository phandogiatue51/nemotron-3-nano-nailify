# Nailify RAG - AI-Powered Nail Design Recommendation System

An intelligent nail design recommendation system powered by Retrieval Augmented Generation (RAG) that provides personalized nail design suggestions based on customer profiles, preferences, and product availability.

## Features

- **AI-Powered Recommendations**: Uses LLM (Ollama or OpenRouter) to generate personalized nail design recommendations
- **Semantic Search**: Leverages ChromaDB for intelligent component retrieval based on semantic similarity
- **Customer Profiling**: Considers multiple customer attributes:
  - Skin tone and shade
  - Hand shape
  - Occupation
  - Nail condition
  - Preferred colors, styles, and occasions
- **Flexible LLM Support**: Supports both local Ollama and OpenRouter cloud-based models
- **FastAPI Backend**: RESTful API for integration with frontend applications
- **Vector Database**: Persistent ChromaDB storage for efficient similarity searches

## Project Structure

```
NailRandomize/
├── main.py                 # Main CLI application
├── api.py                  # FastAPI server
├── config.yaml             # Configuration file
├── requirements.txt        # Project dependencies
├── sync_api_data.py        # Data synchronization utility
├── data/
│   └── vector_db/          # ChromaDB persistent storage
├── notebooks/
│   └── testing.ipynb       # Jupyter notebook for testing
└── src/
    ├── api_client.py       # External API client
    ├── rag_engine.py       # Core RAG recommendation engine
    ├── vector_store.py     # ChromaDB wrapper and management
    ├── prompt_templates.py # LLM prompt templates
    ├── utils.py            # Utility functions
    └── __init__.py
```

## Prerequisites

- Python 3.8+
- Ollama (for local LLM) or OpenRouter API key (for cloud LLM)
- pip (Python package manager)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd NailRandomize
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create `.env` file):
   ```env
   # LLM Provider: 'ollama' or 'openrouter'
   LLM_PROVIDER=ollama
   
   # For Ollama (local):
   OLLAMA_GENERATE_URL=http://localhost:11434/api/generate
   
   # For OpenRouter (cloud):
   OPENROUTER_API_KEY=your_api_key_here
   OPENROUTER_MODEL=deepseek/deepseek-chat
   ```

## Configuration

Edit `config.yaml` to customize:

```yaml
ollama:
  model: "deepseek-r1:1.5b"      # LLM model
  temperature: 0.3               # Creativity level
  top_p: 0.9                      # Nucleus sampling
  max_tokens: 2000                # Response length

chromadb:
  persist_directory: "./data/vector_db"
  collection_name: "nail_components"

rag:
  top_k: 3                        # Number of components to retrieve
  similarity_threshold: 0.7       # Minimum similarity score
```

## Usage

### Running the Main Application

```bash
python main.py
```

This will:
- Display available products (shapes, surfaces, components)
- Fetch customer profile data
- Generate personalized nail design recommendations

### Running the FastAPI Server

```bash
python api.py
```

The API will be available at `http://localhost:8000` with OpenAPI documentation at `/docs`.

**Example API Request**:
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d {
    "skinTone": "Warm",
    "skinShade": "Medium",
    "handShape": "Long",
    "occupation": "Software Engineer",
    "nailCondition": "Healthy",
    "preferredColors": ["Red", "Blue"],
    "preferredStyles": ["Minimalist"],
    "preferredOccasions": ["Work", "Casual"]
  }
```

### Syncing API Data

```bash
python sync_api_data.py
```

Updates the vector database with the latest product components from the API.

## Architecture

### RAG Engine
The `NailRAGEngine` implements a retrieval-augmented generation pipeline:
1. **Query Building**: Constructs semantic search queries from customer data
2. **Retrieval**: Searches ChromaDB for relevant nail components
3. **Augmentation**: Formats retrieved components as context
4. **Generation**: Uses LLM to create personalized recommendations

### Vector Store
- **Embedding Model**: Uses `sentence-transformers` (all-MiniLM-L6-v2) for encoding
- **Similarity Metric**: Cosine similarity for component matching
- **Persistence**: ChromaDB handles data persistence across sessions
