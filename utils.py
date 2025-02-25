import streamlit as st

def initialize_session_state():
    """Initialize session state variables if they are not already initialized."""
    
    # store conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # store chunks from the processed document
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = None

    # store embeddings for document chunks
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = None

    # store user query and chatbot response if needed
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""  
    
    if "response" not in st.session_state:
        st.session_state.response = ""  
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    # for handling session state reset flag (for example, when a new document is uploaded)
    if "reset_session" in st.session_state and st.session_state.reset_session:
        reset_session()

def reset_session():
    """Reset session state variables to their default values."""
    st.session_state.messages = []
    st.session_state.document_chunks = None
    st.session_state.document_embeddings = None
    st.session_state.user_query = ""
    st.session_state.response = ""
    st.session_state.reset_session = False

    # experimental_rerun()
    st.rerun()
