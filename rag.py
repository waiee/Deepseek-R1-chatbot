from typing import List, Dict
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def read_file(file_path: str) -> str:
    """Read different file formats and return text content."""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ' '.join([page.extract_text() for page in pdf_reader.pages])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:  # Assume txt file
            # Try reading with 'utf-8' first
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            except UnicodeDecodeError:
                # If utf-8 fails, try 'latin-1' or 'ISO-8859-1'
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
    except Exception as e:
        text = ""
        print(f"Error reading file {file_path}: {e}")
    return text


def process_document(file_path: str, chunk_size: int = 1000) -> List[str]:
    """Process document and split into chunks."""
    text = read_file(file_path)
    
    if not text:
        return []  # Return empty list if no text was read
    
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
    """Create embeddings for document chunks using TF-IDF."""
    if not chunks:
        return {}
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the document chunks into a document-term matrix
    embeddings = vectorizer.fit_transform(chunks)
    
    # Return embeddings and the vectorizer for later use
    return {'embeddings': embeddings, 'vectorizer': vectorizer}

def get_best_matching_chunk(query: str, chunks: List[str], embeddings: Dict) -> str:
    """Find the chunk most relevant to the query using cosine similarity."""
    if not embeddings:
        return ""
    
    # Transform the query into a vector using the same vectorizer
    query_embedding = embeddings['vectorizer'].transform([query])
    
    # Calculate cosine similarity between the query and document chunks
    similarities = cosine_similarity(query_embedding, embeddings['embeddings'])
    
    # Find the index of the chunk with the highest similarity
    best_match_index = similarities.argmax()
    return chunks[best_match_index]

def find_relevant_chunks(query: str, chunks: List[str], embeddings: Dict, top_k: int = 3) -> List[str]:
    """Find top K relevant chunks based on the query."""
    if not embeddings:
        return []
    
    # Transform the query into a vector using the same vectorizer
    query_embedding = embeddings['vectorizer'].transform([query])
    
    # Calculate cosine similarity between the query and document chunks
    similarities = cosine_similarity(query_embedding, embeddings['embeddings']).flatten()
    
    # Get top K chunks
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]
