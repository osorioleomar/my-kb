# RAG Application

A CPU-friendly Retrieval-Augmented Generation (RAG) application that processes text files, creates embeddings, and provides a search interface. This project is designed to work efficiently on machines without GPU acceleration.

## Features

- Text chunking with metadata preservation
- CPU-optimized embeddings using sentence-transformers
- FAISS vector database for efficient similarity search
- Modern web interface for searching and summarizing content
- Support for processing multiple text files
- Progress tracking and error handling
- Markdown rendering for better readability
- Real-time search and summarization using Groq API

## Technologies Used

- **Python 3.x**
- **sentence-transformers**: For generating text embeddings
- **faiss-cpu**: For vector storage and similarity search
- **tqdm**: For progress tracking
- **python-dotenv**: For configuration management
- **numpy & pandas**: For data processing
- **TailwindCSS**: For modern UI styling
- **Marked.js**: For markdown rendering
- **Groq API**: For LLM-powered summarization

## Project Structure

```
my-kb/
├── data/
│   ├── raw/           # Original text files
│   └── processed/     # Processed JSON chunks
├── src/
│   ├── chunk.py       # Text chunking script
│   ├── embed.py       # Embedding generation script
│   └── api.py         # FastAPI server for vector search
├── index.html         # Main web interface for search and summarization
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/my-kb.git
   cd my-kb
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Groq API key:
   - Get an API key from [Groq](https://console.groq.com)
   - Update the API key in `index.html`

## Usage

### 1. Prepare Your Data
Place your `.txt` files in the `data/raw` directory.

### 2. Process Text Files
Run the chunking script to process your text files:
```bash
python src/chunk.py
```
This will:
- Split text into chunks with metadata
- Save processed chunks as JSON files in `data/processed`

### 3. Generate Embeddings
Create embeddings for your text chunks:
```bash
python src/embed.py
```
This will:
- Generate embeddings using sentence-transformers
- Store them in a FAISS vector database

### 4. Start the Application
1. Start the backend API server:
   ```bash
   python src/api.py
   ```
   This will start the FastAPI server that handles vector search requests.

2. Open `index.html` in your web browser to use the search interface.

The application is now ready to use! You can:
- Enter queries in the search box
- Get real-time search results and summaries
- View source documents and similarity scores
- See markdown-formatted responses

## Technical Details

### Chunking Strategy
- Chunk size: 512 tokens
- Overlap: 128 tokens
- Preserves paragraph boundaries
- Includes metadata (source file, position, timestamp)

### Embedding Model
- Model: 'all-MiniLM-L6-v2'
- Dimension: 384
- Optimized for CPU usage

### Search Interface
- Modern UI built with TailwindCSS
- Real-time search using FAISS
- LLM-powered summarization using Groq API
- Markdown rendering for better readability
- Responsive design for all devices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This project is under active development. Features and documentation will be updated as the project progresses.* 