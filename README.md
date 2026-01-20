# RAG Chatbot with Milvus

A Retrieval-Augmented Generation (RAG) chatbot that scrapes 5 web pages, stores them in Milvus vector database, and answers questions ONLY based on those specific pages.

## Features

- ✅ Web scraping from 5 specified URLs
- ✅ Text chunking and processing
- ✅ Vector embeddings using Sentence Transformers
- ✅ Milvus vector database for storage
- ✅ RAG-based chatbot with OpenAI GPT
- ✅ Restricted responses (only answers from stored documents)

## Project Structure

```
├── main.py              # Main pipeline to scrape and store data
├── scrapper.py          # Web scraping functionality
├── text_processor.py    # Text cleaning and chunking
├── milvus_manager.py    # Milvus database operations
├── chatbot.py           # RAG chatbot implementation
├── pagesurl.txt         # URLs to scrape (one per line)
├── scraped_pages/       # Saved text files (auto-generated)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## Prerequisites

1. **Python 3.8+**
2. **Milvus** - Vector database
   - Install via Docker: 
     ```bash
     wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
     docker-compose up -d
     ```
   - Or use [Milvus Lite](https://milvus.io/docs/milvus_lite.md) for development
3. **OpenAI API Key** - For the chatbot

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   ```

4. **Start Milvus** (if using Docker):
   ```bash
   docker-compose up -d
   ```

## Usage

### Stpagesurl.txt` and add your URLs (one per line):

```
https://your-first-page.com
https://your-second-page.com
https://your-third-page.com
https://your-fourth-page.com
https://your-fifth-page.com
```

The file currently contains 5 neosapients.ai pages. "https://your-fifth-page.com"
]
```

### Step 2: Run the Setup Pipeline

This will scrape the pages, process the text, and store it in Milvus:

```bash
python main.py
```

Expected output:
- Scrapes pages from `pagesurl.txt`
- Saves each page as a separate .txt file in `scraped_pages/` directory
- Chunks the text into manageable pieces
- Generates embeddings
- Stores everything in Milvus
- Runs a test search

### Step 3: Start the Chatbot

```bash
python chatbot.py
```

Now you can ask questions! The chatbot will:
- ✅ Answer questions based ONLY on your 5 pages
- ✅ Cite sources for its answers
- ✅ Refuse to answer questions outside its knowledge base

Example conversation:
```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...

Sources:
- Machine learning (https://en.wikipedia.org/wiki/Machine_learning)

You: Who won the 2024 Olympics?
Assistant: I don't have information about that in my knowledge base.
```

## How It Works

1. **Web Scraping** (`scrapper.py`)
   - Uses BeautifulSoup to extract text from web pages
   - Cleans and formats the content

2. **Text Processing** (`text_processor.py`)
   - Splits documents into overlapping chunks (default: 500 chars)
   - Preserves context with chunk overlap (default: 50 chars)

3. **Vector Storage** (`milvus_manager.py`)
   - Generates embeddings using `all-MiniLM-L6-v2` model
   - Stores embeddings + metadata in Milvus
   - Creates indexes for fast similarity search

4. **RAG Chatbot** (`chatbot.py`)
   - Converts user queries to embeddings
   - Searches Milvus for similar document chunks
   - Checks relevance threshold
   - Uses OpenAI GPT with strict prompting to answer ONLY from retrieved context
   - Cites sources

## Configuration

### Chunk Settings
In `main.py`, adjust chunk parameters:
```python
chunker = TextChunker(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

### Search Settings
In `chatbot.py`, adjust retrieval:
```python
response = self.generate_response(query, top_k=5)  # Number of chunks to retrieve
```

### Relevance Threshold
In `chatbot.py` line 19:
```python
return best_score > 0.3  # Adjust threshold (0.0-1.0)
```

## Troubleshooting

### Milvus Connection Error
- Ensure Milvus is running: `docker ps`
- Check host/port in `.env`

### OpenAI API Error
- Verify API key in `.env`
- Check API quota/billing

### Scraping Fails
- Some websites block scrapers
- Try different URLs or adjust headers in `scrapper.py`

## Customization

### Use Different Embedding Model
In `milvus_manager.py` line 15:
```python
self.encoder = SentenceTransformer('all-mpnet-base-v2')  # Better quality
self.embedding_dim = 768  # Update dimension accordingly
```

### Use Different LLM
Replace OpenAI in `chatbot.py` with:
- Anthropic Claude
- Local models (Ollama, LM Studio)
- Azure OpenAI

### Add More Pages
Just add more URLs to the `urls` list in `main.py` (not limited to 5)

## License

MIT

## Notes

- First run downloads the embedding model (~80MB)
- Milvus data persists in Docker volumes
- To reset: Drop collection or restart Milvus with `docker-compose down -v`
