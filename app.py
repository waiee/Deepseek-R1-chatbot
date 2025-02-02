import streamlit as st
from rag import process_document, create_embeddings
from chat_module import get_chat_response
from utils import initialize_session_state
import tempfile

def main():
    st.title("Deepseek-R1 Chat Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader("Upload your document here", type=["txt", "pdf", "docx"])
        
        # Reset session state when a new document is uploaded
        if uploaded_file:
            # Reset session state to avoid conflicts with new document uploads
            if "reset_session" not in st.session_state or st.session_state.reset_session:
                st.session_state.reset_session = False
                initialize_session_state()

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process document and create embeddings
            chunks = process_document(tmp_file_path)
            embeddings = create_embeddings(chunks)
            print(embeddings.keys()) 

            
            # Store in session state
            st.session_state.document_chunks = chunks
            st.session_state.document_embeddings = embeddings
            st.success("Document processed successfully!")
        
        # Reset session button
        if st.button("Reset Session"):
            st.session_state.reset_session = True
            initialize_session_state()
            st.experimental_rerun()
    
    # Chat interface
    # st.header("Your AI-Assistant")
    
    # Display chat history
    if "messages" in st.session_state and st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if hasattr(st.session_state, 'document_embeddings') and st.session_state.document_embeddings:
            # Get bot response using the document context
            response = get_chat_response(
                prompt, 
                st.session_state.document_chunks,
                st.session_state.document_embeddings
            )
        else:
            # If no document context is available, answer based on general knowledge
            response = get_chat_response(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
