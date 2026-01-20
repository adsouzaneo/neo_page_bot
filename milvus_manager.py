from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os


class MilvusManager:
    """Manages Milvus vector database operations"""
    
    def __init__(self, collection_name: str = "rag_documents", 
                 host: str = "localhost", port: str = "19530"):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
    def connect(self):
        """Connect to Milvus server"""
        try:
            # Use Milvus Lite (embedded mode)
            connections.connect(
                alias="default",
                uri="./milvus_demo.db"
            )
            print(f"Connected to Milvus Lite (embedded mode)")
        except Exception as e:
            print(f"Error connecting to Milvus: {str(e)}")
            raise
    
    def create_collection(self):
        """Create a collection for storing document embeddings"""
        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Dropped existing collection: {self.collection_name}")
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_index", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields=fields, description="RAG document collection")
        
        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Created collection: {self.collection_name}")
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print("Created index on embedding field")
        
    def load_collection(self):
        """Load collection into memory"""
        if not self.collection:
            self.collection = Collection(self.collection_name)
        self.collection.load()
        print(f"Loaded collection: {self.collection_name}")
    
    def insert_documents(self, chunks: List[Dict]):
        """Insert document chunks into Milvus"""
        if not chunks:
            print("No chunks to insert")
            return
        
        # Prepare data
        texts = [chunk['text'] for chunk in chunks]
        urls = [chunk['metadata']['url'] for chunk in chunks]
        titles = [chunk['metadata']['title'] for chunk in chunks]
        chunk_indices = [chunk['metadata']['chunk_index'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Insert data
        entities = [
            embeddings.tolist(),
            texts,
            urls,
            titles,
            chunk_indices
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
        print(f"Inserted {len(texts)} chunks into Milvus")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.encoder.encode([query])[0]
        
        # Search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "url", "title", "chunk_index"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'text': hit.entity.get('text'),
                    'url': hit.entity.get('url'),
                    'title': hit.entity.get('title'),
                    'chunk_index': hit.entity.get('chunk_index'),
                    'score': hit.score
                })
        
        return formatted_results
    
    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("Disconnected from Milvus")
