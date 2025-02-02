from typing import List, Dict
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer  # More advanced embedding model

def read_file(file_path: str) -> str:
    """Read different file formats and return text content."""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ' '.join([page.extract_text() for page in pdf_reader.pages])
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    else:  # Assume txt file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try a different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin1') as file:
                text = file.read()
    return text

def process_document(file_path: str, chunk_size: int = 1000) -> List[str]:
    """Process document and split into chunks."""
    text = read_file(file_path)
    
    # Clean text if necessary (remove extra spaces, line breaks, etc.)
    text = ' '.join(text.split())
    
    # Split the text into chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def create_embeddings(chunks: List[str]) -> Dict:
    """Create embeddings for document chunks using Sentence-BERT."""
    # Initialize a pre-trained Sentence-BERT model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight model suitable for semantic tasks
    embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # Return embeddings and model for later use
    return {
        'embeddings': embeddings
    }

def get_best_matching_chunk(query: str, chunks: List[str], embeddings: Dict) -> str:
    """Find the chunk most relevant to the query using cosine similarity."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Calculate cosine similarity between the query and document chunks
    similarities = cosine_similarity(query_embedding, embeddings['embeddings'])
    
    # Find the index of the chunk with the highest similarity
    best_match_index = similarities.argmax()
    return chunks[best_match_index]

def find_relevant_chunks(query: str, chunks: List[str], embeddings: Dict, top_k: int = 3) -> List[str]:
    """Find top K relevant chunks based on the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Calculate cosine similarity between the query and document chunks
    similarities = cosine_similarity(query_embedding, embeddings['embeddings']).flatten()
    
    # Get top K chunks
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]
