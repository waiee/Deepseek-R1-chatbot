# utils.py
import streamlit as st

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = None
        
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = None