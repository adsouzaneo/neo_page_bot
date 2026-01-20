"""
Streamlit Chatbot Interface for RAG System
Interactive UI for asking questions about the scraped NeoSapients pages
"""

import streamlit as st
from milvus_manager import MilvusManager
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="NeoSapients RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #1a1a1a;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #43a047;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #e65100;
        border-left: 3px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "milvus" not in st.session_state:
    st.session_state.milvus = None

# Initialize Milvus
@st.cache_resource
def init_milvus():
    """Initialize Milvus connection"""
    milvus = MilvusManager()
    milvus.connect()
    milvus.load_collection()
    return milvus

# Query function
def query_rag(question: str, use_openai: bool = True, top_k: int = 5):
    """Query the RAG system"""
    milvus = st.session_state.milvus
    
    # Retrieve relevant documents
    results = milvus.search(question, top_k=top_k)
    
    print(f"DEBUG: Query: '{question}'")
    print(f"DEBUG: Results count: {len(results) if results else 0}")
    
    if not results:
        return {
            "answer": "I don't have any information to answer that question. I can only answer questions based on the specific documents I was trained on.",
            "sources": [],
            "relevance": 0.0
        }
    
    # Check relevance
    best_score = results[0]['score']
    print(f"DEBUG: Best score: {best_score}")
    print(f"DEBUG: Top result: {results[0]['title']}")
    
    if best_score < 0.25:
        return {
            "answer": "I don't have any information to answer that question. I can only answer questions based on the specific documents I was trained on.",
            "sources": [],
            "relevance": best_score
        }
    
    # Build context
    context = "\n\n".join([
        f"Source: {doc['title']} ({doc['url']})\n{doc['text']}"
        for doc in results
    ])
    
    # Generate answer
    if use_openai and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            system_prompt = """You are a helpful assistant that answers questions based on the provided context from NeoSapients documentation.

RULES:
1. Use ONLY the information from the provided context to answer questions
2. If the context contains relevant information, provide a helpful answer
3. If the user asks vague questions like "the terms and conditions" or "what are the terms", provide a summary of the key points from the terms of use
4. Be helpful and informative
5. If the context truly doesn't contain information to answer the question, say "I don't have information about that in my knowledge base"
6. Be concise but complete"""

            user_prompt = f"""Context from NeoSapients knowledge base:
{context}

Question: {question}

Provide a helpful answer based on the context above."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response: {str(e)}\n\nHere's the retrieved context:\n{context[:500]}..."
    else:
        # Fallback: just show the retrieved context
        answer = f"**Retrieved Information:**\n\n{results[0]['text']}\n\n*Note: Add OpenAI API key to .env for AI-generated answers*"
    
    # Format sources
    sources = [
        {
            "title": doc['title'],
            "url": doc['url'],
            "score": doc['score']
        }
        for doc in results[:3]
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "relevance": best_score
    }

# Main app
def main():
    # Header
    st.title("🤖 NeoSapients RAG Chatbot")
    st.markdown("Ask questions about NeoSapients, Context OS, and their AI solutions")
    
    # Initialize Milvus
    if st.session_state.milvus is None:
        with st.spinner("Connecting to knowledge base..."):
            st.session_state.milvus = init_milvus()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            st.success("✓ OpenAI API key configured")
            use_openai = True
        else:
            st.warning("⚠️ OpenAI API key not set")
            st.info("Add your API key to .env file for AI-powered answers")
            use_openai = False
        
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
        
        st.divider()
        
        st.header("📚 Knowledge Base")
        st.write("6 pages from NeoSapients")
        
        st.divider()
        
        st.header("💡 Example Questions")
        example_questions = [
            "What is NeoSapients?",
            "What is Context OS?",
            "Tell me about AI agents",
            "What industries do you serve?",
            "What career opportunities are available?",
        ]
        
        for question in example_questions:
            if st.button(question, key=f"ex_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>🧑 You:</b><br>
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <b>🤖 Assistant:</b><br>
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
    
    # Input box at the bottom
    st.divider()
    
    # Create columns for input and button
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question:",
            key="user_input",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process question
    if send_button and user_question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Get response
        with st.spinner("Thinking..."):
            result = query_rag(user_question, use_openai=use_openai, top_k=top_k)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
        
        # Rerun to update chat
        st.rerun()

if __name__ == "__main__":
    main()
