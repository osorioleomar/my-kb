# RAG Application Plan

## Overview
This project will create a simple RAG (Retrieval-Augmented Generation) application that can run on CPU-only machines. The application will process text files, create embeddings, and provide a search interface.

## Project Structure
```
my-kb/
├── data/
│   ├── raw/           # Original text files
│   └── processed/     # Processed JSON chunks
├── src/
│   ├── chunk.py       # Text chunking script
│   ├── embed.py       # Embedding generation script
│   └── app.py         # Streamlit UI application
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Components

### 1. Text Chunking (chunk.py) ✅
- Input: Folder containing text files
- Output: JSON files with chunks and metadata
- Features:
  - Read text files from specified folder
  - Split text into chunks using sliding window approach
  - Add metadata to each chunk:
    - Source file name
    - Chunk ID
    - Position in document
    - Timestamp
    - Word count
  - Save chunks as JSON files in processed folder
  - Handle different text file encodings
  - Progress tracking

### 2. Embedding Generation (embed.py) ✅
- Input: Folder containing JSON chunk files
- Output: Vector database with embeddings
- Features:
  - Use sentence-transformers for CPU-friendly embeddings
  - Process chunks from JSON files
  - Store embeddings in FAISS vector database
  - Save vector database to disk
  - Handle large datasets efficiently
  - Progress tracking

### 3. Search Interface (app.py)
- Input: Vector database
- Output: Interactive web interface
- Features:
  - Streamlit-based UI
  - Search input field
  - Display results with:
    - Source file
    - Chunk content
    - Similarity score
    - Context (surrounding chunks)
  - Filter results by source file
  - Adjust number of results
  - Clear and intuitive design

## Technical Details

### Dependencies
- sentence-transformers: For generating embeddings
- faiss-cpu: For vector storage and search
- streamlit: For the web interface
- tqdm: For progress tracking
- python-dotenv: For configuration

### Chunking Strategy
- Use sliding window approach
- Chunk size: 512 tokens
- Overlap: 128 tokens
- Preserve paragraph boundaries when possible

### Embedding Model
- Use 'all-MiniLM-L6-v2' from sentence-transformers
- 384-dimensional embeddings
- Optimized for CPU usage

### Vector Database
- FAISS IndexFlatL2 for exact search
- Support for incremental updates
- Persistence to disk

## Implementation Steps

1. **Setup Project Structure** ✅
   - Create directory structure ✅
   - Initialize git repository
   - Create requirements.txt ✅

2. **Implement Chunking Script** ✅
   - Create chunk.py ✅
   - Implement text processing ✅
   - Add metadata generation ✅
   - Test with sample files ✅

3. **Implement Embedding Script** ✅
   - Create embed.py ✅
   - Set up sentence-transformers ✅
   - Implement FAISS integration ✅
   - Test with sample chunks ✅

4. **Create Search Interface**
   - Create app.py
   - Design Streamlit UI
   - Implement search functionality
   - Add result display

5. **Testing and Optimization**
   - Test with various text files ✅
   - Optimize chunking parameters ✅
   - Profile performance
   - Add error handling

6. **Documentation**
   - Create README.md
   - Add usage instructions
   - Document API
   - Add examples

## Future Enhancements
- Add support for more file formats
- Implement batch processing
- Add caching for better performance
- Support for custom embedding models
- Add result highlighting
- Implement result ranking options 