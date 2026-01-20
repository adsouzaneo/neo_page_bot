from typing import List, Dict
import re


class TextChunker:
    """Splits text into chunks for embedding"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks"""
        text = self.clean_text(text)
        
        # Split by sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': metadata.copy()
                })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': metadata.copy()
            })
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process multiple documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            metadata = {
                'url': doc['url'],
                'title': doc['title']
            }
            chunks = self.chunk_text(doc['content'], metadata)
            
            # Add chunk index to metadata
            for idx, chunk in enumerate(chunks):
                chunk['metadata']['chunk_index'] = idx
                all_chunks.append(chunk)
        
        return all_chunks
