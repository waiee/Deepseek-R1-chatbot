import streamlit as st
from rag import process_document, create_embeddings
from chat_module import get_chat_response
from utils import initialize_session_state
import tempfile

def main():
    st.title("Deepseek-R1 Chat Assistant")
    
    initialize_session_state()
    
    # sidebar doc upload
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader("Upload your document here", type=["txt", "pdf", "docx"])
        
        # reset session state
        if uploaded_file:
            # reset session state to avoid conflicts
            if "reset_session" not in st.session_state or st.session_state.reset_session:
                st.session_state.reset_session = False
                initialize_session_state()

            # temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # process document and create embeddings
            chunks = process_document(tmp_file_path)
            embeddings = create_embeddings(chunks)
            print(embeddings.keys()) 

            st.session_state.document_chunks = chunks
            st.session_state.document_embeddings = embeddings
            st.success("Document processed successfully!")
        
        if st.button("Reset Session"):
            st.session_state.reset_session = True
            initialize_session_state()
            st.experimental_rerun()
    
    ### Chat interface ###
    
    # chat history
    if "messages" in st.session_state and st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if hasattr(st.session_state, 'document_embeddings') and st.session_state.document_embeddings:
            response = get_chat_response(
                prompt, 
                st.session_state.document_chunks,
                st.session_state.document_embeddings
            )
        else:
            response = get_chat_response(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
