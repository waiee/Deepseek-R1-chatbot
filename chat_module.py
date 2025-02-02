# chat_module.py
from typing import List, Dict, Optional
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ChatBot:
    def __init__(self):
        # Initialize Ollama API settings
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "gemma:7b"
        
    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            }
        }
        
        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Ollama API response: {str(e)}"

def find_relevant_chunks(query: str, chunks: List[str], embeddings: Dict, top_k: int = 3) -> List[str]:
    """Find most relevant document chunks for the query."""
    # Create query embedding
    query_embedding = embeddings['vectorizer'].transform([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, embeddings['embeddings']).flatten()
    
    # Get top k chunks
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]

def get_chat_response(query: str, chunks: Optional[List[str]] = None, embeddings: Optional[Dict] = None) -> str:
    """Get chatbot response, incorporating document context if available."""
    chatbot = ChatBot()
    
    if chunks and embeddings:
        # Find relevant document chunks
        relevant_chunks = find_relevant_chunks(query, chunks, embeddings)
        
        # Create context-aware prompt
        context = "\n".join(relevant_chunks)
        prompt = f"""Based on the following context, please respond to the query.

Context:
{context}

Query: {query}

Please provide a response that incorporates relevant information from the context while maintaining a natural conversational tone."""
    else:
        # If no document context is available, use the query directly
        prompt = query
    
    # Generate response
    return chatbot.generate_response(prompt)