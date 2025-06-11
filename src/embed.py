import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator with a sentence transformer model.
        Args:
            model_name (str): Name of the sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def load_chunks(self, json_file: Path) -> List[Dict]:
        """Load chunks from a JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['chunks']

    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for a list of chunks."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create a FAISS index for the embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def process_file(self, json_file: Path, output_dir: Path) -> None:
        """Process a single JSON file and save its embeddings."""
        try:
            chunks = self.load_chunks(json_file)
            embeddings = self.generate_embeddings(chunks)
            index = self.create_faiss_index(embeddings)
            output_file = output_dir / f"{json_file.stem}_index.faiss"
            faiss.write_index(index, str(output_file))
            print(f"Embeddings saved to {output_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

def main():
    input_dir = Path("data/processed")
    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = EmbeddingGenerator()
    json_files = list(input_dir.glob("*_chunks.json"))
    if not json_files:
        print(f"No JSON chunk files found in {input_dir}")
        return
    print(f"Found {len(json_files)} JSON files to process")
    for json_file in tqdm(json_files, desc="Processing files"):
        generator.process_file(json_file, output_dir)
    print(f"Processing complete. Embeddings saved in {output_dir}")

if __name__ == "__main__":
    main() 