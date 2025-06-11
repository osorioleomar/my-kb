import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import re

class TextChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        """
        Initialize the text chunker with specified chunk size and overlap.
        
        Args:
            chunk_size (int): Number of tokens per chunk
            overlap (int): Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words (simple whitespace split)."""
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into a string."""
        return ' '.join(tokens)

    def create_chunks(self, text: str, source_file: str) -> List[Dict]:
        """Create chunks from tokens with sliding window and metadata."""
        tokens = self.tokenize(text)
        chunks = []
        chunk_id = 0
        step = self.chunk_size - self.overlap
        for start in range(0, len(tokens), step):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunk_text = self.detokenize(chunk_tokens)
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'source_file': source_file,
                'position': start,
                'word_count': len(chunk_tokens),
                'metadata': {
                    'chunk_size': self.chunk_size,
                    'overlap': self.overlap
                }
            })
            chunk_id += 1
        return chunks

    def process_file(self, file_path: Path, output_dir: Path) -> None:
        """Process a single file and save its chunks."""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Create chunks
            chunks = self.create_chunks(text, file_path.name)
            
            # Save chunks to JSON
            output_file = output_dir / f"{file_path.stem}_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_name': file_path.name,
                    'total_chunks': len(chunks),
                    'chunks': chunks
                }, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    # Get input and output directories
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize chunker with smaller chunk size
    chunker = TextChunker(chunk_size=200, overlap=50)
    
    # Process all txt files
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} text files to process")
    for file_path in tqdm(txt_files, desc="Processing files"):
        chunker.process_file(file_path, output_dir)
    
    print(f"Processing complete. Chunks saved in {output_dir}")

if __name__ == "__main__":
    main() 