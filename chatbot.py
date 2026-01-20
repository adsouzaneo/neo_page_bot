from milvus_manager import MilvusManager
from openai import OpenAI
import os
from dotenv import load_dotenv


class RAGChatbot:
    """RAG-based chatbot that only answers based on stored documents"""
    
    def __init__(self, milvus_manager: MilvusManager, api_key: str = None):
        self.milvus = milvus_manager
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
    def is_relevant_query(self, query: str, context_docs: list) -> bool:
        """Check if the query is relevant to the retrieved documents"""
        # Simple relevance check based on minimum similarity score
        if not context_docs:
            return False
        
        # If the best match has a score below threshold, consider irrelevant
        best_score = max([doc['score'] for doc in context_docs])
        return best_score > 0.3  # Threshold can be adjusted
    
    def generate_response(self, query: str, top_k: int = 5) -> str:
        """Generate a response based on retrieved documents"""
        # Retrieve relevant documents
        results = self.milvus.search(query, top_k=top_k)
        
        if not results:
            return "I don't have any information to answer that question. I can only answer questions based on the specific documents I was trained on."
        
        # Check relevance
        if not self.is_relevant_query(query, results):
            return "I don't have any information to answer that question. I can only answer questions based on the specific documents I was trained on."
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc['title']} ({doc['url']})\n{doc['text']}"
            for doc in results
        ])
        
        # Create prompt with strict instructions
        system_prompt = """You are a helpful assistant that ONLY answers questions based on the provided context.
        
CRITICAL RULES:
1. ONLY use information from the provided context to answer questions
2. If the context doesn't contain information to answer the question, you MUST say "I don't have information about that in my knowledge base"
3. Do NOT use your general knowledge or training data
4. Always cite which source document you're using
5. Be concise and accurate
6. If you're unsure, say so rather than making assumptions"""

        user_prompt = f"""Context from knowledge base:
{context}

Question: {query}

Answer based ONLY on the context above. If the context doesn't contain relevant information, say so."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Add source references
            sources = list(set([f"{doc['title']} ({doc['url']})" for doc in results[:3]]))
            sources_text = "\n\nSources:\n" + "\n".join([f"- {s}" for s in sources])
            
            return answer + sources_text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self):
        """Interactive chat interface"""
        print("=" * 70)
        print("RAG Chatbot - Ask questions about the stored documents")
        print("Type 'quit' or 'exit' to end the conversation")
        print("=" * 70)
        
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nAssistant: ", end="")
            response = self.generate_response(query)
            print(response)


if __name__ == "__main__":
    load_dotenv()
    
    # Initialize Milvus manager
    milvus = MilvusManager()
    milvus.connect()
    milvus.load_collection()
    
    # Create chatbot
    chatbot = RAGChatbot(milvus)
    
    # Start chat
    chatbot.chat()
    
    # Cleanup
    milvus.disconnect()
