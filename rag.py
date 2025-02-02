#rag.py

from typing import List, Dict
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    return text

def process_document(file_path: str, chunk_size: int = 1000) -> List[str]:
    """Process document and split into chunks."""
    # Read the document
    text = read_file(file_path)
    
    # Simple chunking by words
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def create_embeddings(chunks: List[str]) -> Dict:
    """Create TF-IDF embeddings for document chunks."""
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Create embeddings
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    return {
        'vectorizer': vectorizer,
        'embeddings': tfidf_matrix,
        'vocabulary': vectorizer.vocabulary_
    }