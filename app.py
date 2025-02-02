import streamlit as st
from rag import process_document, create_embeddings
from chat_module import get_chat_response
from utils import initialize_session_state
import tempfile
import re  # For regular expression

def main():
    st.title("Deepseek-R1 Chat Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload your document here", type=["txt", "pdf", "docx"])
        
        if uploaded_file:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process document and create embeddings
            chunks = process_document(tmp_file_path)
            embeddings = create_embeddings(chunks)
            
            # Store in session state
            st.session_state.document_chunks = chunks
            st.session_state.document_embeddings = embeddings
            st.success("Document processed successfully!")
    
    # Chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            response = get_chat_response(
                prompt, 
                st.session_state.document_chunks if hasattr(st.session_state, 'document_chunks') else None,
                st.session_state.document_embeddings if hasattr(st.session_state, 'document_embeddings') else None
            )
            
            # Remove <think> tags from response
            cleaned_response = re.sub(r'<think>.*?</think>', '', response)
            
            st.markdown(cleaned_response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

if __name__ == "__main__":
    main()
