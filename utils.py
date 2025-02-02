import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

def initialize_session_state():
    """Initialize session state variables if they are not already initialized."""
    
    # Initialize messages (store conversation history)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize document chunks (store chunks from the processed document)
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = None

    # Initialize document embeddings (store embeddings for document chunks)
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = None

    # Store user query and chatbot response if needed
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""  
    
    if "response" not in st.session_state:
        st.session_state.response = ""  
    
    # Optionally: Prevent re-initialization if already set
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    # Handle session state reset flag (for example, when a new document is uploaded)
    if "reset_session" in st.session_state and st.session_state.reset_session:
        reset_session()

def reset_session():
    """Reset session state variables to their default values."""
    st.session_state.messages = []
    st.session_state.document_chunks = None
    st.session_state.document_embeddings = None
    st.session_state.user_query = ""
    st.session_state.response = ""
    st.session_state.reset_session = False  # Reset the flag after resetting

def chunk_and_embed_document(text: str, chunk_size: int = 1000):
    """Chunk document text and create embeddings using TF-IDF."""
    # Split text into chunks of the specified size
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # Use TF-IDF to create embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)
    
    return chunks, embeddings
