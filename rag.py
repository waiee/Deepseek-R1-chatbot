from typing import List, Dict
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import docx

def read_file(file_path: str) -> str:
    """Reads text from PDF, DOCX, or TXT files, using OCR if necessary."""
    text = ""

    try:
        if file_path.endswith('.pdf'):
            # extract normally
            with pdfplumber.open(file_path) as pdf:
                text = " ".join([page.extract_text() or '' for page in pdf.pages if page.extract_text()])
            
            # ocr
            if not text.strip():
                print("⚠️ No text found in PDF. Using OCR...")
                images = convert_from_path(file_path)
                text = " ".join([pytesseract.image_to_string(img) for img in images])

        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])

        else:  # txt file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()

    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")

    print(f"🔍 Extracted text preview:\n{text[:500]}")  # debugging
    return text


def process_document(file_path: str, chunk_size: int = 1000) -> List[str]:
    """Processes document and splits into chunks after OCR, if needed."""
    text = read_file(file_path)

    if not text.strip():
        print("🚨 ERROR: No text extracted from document.")
        return []

    # clean text
    text = " ".join(text.split())

    # chunking
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    print(f"✅ Document processed into {len(chunks)} chunks.")
    return chunks


def create_embeddings(chunks: List[str]) -> Dict:
    """Create embeddings for document chunks using TF-IDF."""
    if not chunks:
        return {}
    
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # fit & transform the document chunks into a document-term matrix
    embeddings = vectorizer.fit_transform(chunks)
    
    # return embeddings,vectorizer
    return {'embeddings': embeddings, 'vectorizer': vectorizer}

def get_best_matching_chunk(query: str, chunks: List[str], embeddings: Dict) -> str:
    """Find the chunk most relevant to the query using cosine similarity."""
    if not embeddings:
        return ""
    
    # transform the query into a vector using the same vectorizer
    query_embedding = embeddings['vectorizer'].transform([query])
    
    #  cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings['embeddings'])
    
    # chunk with the highest similarity
    best_match_index = similarities.argmax()
    return chunks[best_match_index]

def find_relevant_chunks(query: str, chunks: List[str], embeddings: Dict, top_k: int = 3) -> List[str]:
    if not embeddings:
        return []

    query_embedding = embeddings['vectorizer'].transform([query])
    similarities = cosine_similarity(query_embedding, embeddings['embeddings']).flatten()
    top_indices = np.argsort(similarities)[-top_k:]

    selected_chunks = [chunks[i] for i in top_indices] 

    print(f"📌 Retrieved text chunks:\n{selected_chunks[:3]}")

    return selected_chunks


