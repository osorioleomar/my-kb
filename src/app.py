import streamlit as st
import faiss
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Helper to load all indexes and metadata
def load_indexes_and_metadata(embeddings_dir, processed_dir):
    indexes = {}
    metadatas = {}
    
    # Debug: Print absolute paths
    st.write(f"Looking for FAISS indexes in: {Path(embeddings_dir).absolute()}")
    st.write(f"Looking for JSON files in: {Path(processed_dir).absolute()}")
    
    for faiss_file in Path(embeddings_dir).glob("*_index.faiss"):
        base = faiss_file.stem.replace('_index', '')  # This gives us 'sample1_chunks' or 'sample2_chunks'
        json_file = Path(processed_dir) / f"{base}.json"  # Now it will be 'sample1_chunks.json' or 'sample2_chunks.json'
        
        # Debug: Print file paths being processed
        st.write(f"Processing FAISS file: {faiss_file}")
        st.write(f"Looking for JSON file: {json_file}")
        
        if not json_file.exists():
            st.warning(f"JSON file not found: {json_file}")
            continue
            
        try:
            # Load FAISS index
            index = faiss.read_index(str(faiss_file))
            indexes[base] = index
            st.success(f"Successfully loaded FAISS index: {faiss_file}")
            
            # Load metadata
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metadatas[base] = data['chunks']
            st.success(f"Successfully loaded metadata: {json_file}")
        except Exception as e:
            st.error(f"Error loading {faiss_file} or {json_file}: {str(e)}")
            
    return indexes, metadatas

# Search function
def search(query, indexes, metadatas, model, top_k=5, file_filter=None):
    query_emb = model.encode([query])
    results = []
    for base, index in indexes.items():
        if file_filter and base != file_filter:
            continue
        D, I = index.search(query_emb.astype(np.float32), top_k)
        for dist, idx in zip(D[0], I[0]):
            if idx < len(metadatas[base]):
                chunk = metadatas[base][idx]
                # Convert L2 distance to similarity score (0 to 1, where 1 is most similar)
                similarity = 1 / (1 + dist)  # This converts distance to similarity score
                results.append({
                    'source_file': chunk['source_file'],
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'score': float(dist),
                    'similarity': float(similarity)
                })
    # Sort by score (lower is better for L2)
    results = sorted(results, key=lambda x: x['score'])[:top_k]
    return results

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Vector Search Demo", layout="wide")
    st.title("RAG Vector Search Demo")
    st.write("Search your chunked knowledge base using vector similarity.")

    # Initialize session state for model
    if 'model' not in st.session_state:
        with st.spinner("Loading the embedding model..."):
            st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load indexes and metadata
    base_dir = Path(__file__).parent.parent  # Go up one level from src
    embeddings_dir = base_dir / "data/embeddings"
    processed_dir = base_dir / "data/processed"
    
    with st.spinner("Loading indexes and metadata..."):
        indexes, metadatas = load_indexes_and_metadata(embeddings_dir, processed_dir)
    
    if not indexes:
        st.error("No FAISS indexes found. Please run the embedding script first.")
        return

    # Sidebar options
    st.sidebar.title("Search Options")
    file_options = list(indexes.keys())
    file_filter = st.sidebar.selectbox("Filter by file (optional)", [None] + file_options)
    top_k = st.sidebar.slider("Number of results", 1, 10, 5)

    # Main search interface
    query = st.text_input("Enter your search query:")
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            results = search(query, indexes, metadatas, st.session_state.model, top_k=top_k, file_filter=file_filter)
        
        if results:
            st.write(f"Found {len(results)} results:")
            for r in results:
                with st.expander(f"File: {r['source_file']} | Similarity: {r['similarity']:.2%}"):
                    st.write(r['text'])
                    st.caption(f"Raw distance score: {r['score']:.4f} (lower is better)")
        else:
            st.info("No results found.")

if __name__ == "__main__":
    main() 