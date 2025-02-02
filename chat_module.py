from typing import List, Dict, Optional
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

class ChatBot:
    def __init__(self):
        # Initialize Ollama API settings
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "deepseek-r1"
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Reuse the same model
        
    def clean_response(self, text: str) -> str:
        """Clean the response by removing thinking patterns."""
        # Remove <think> tags and their content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove alternative thinking patterns
        patterns = [
            r'\[thinking\].*?\[/thinking\]',
            r'\(thinking:.*?\)',
            r'Think:.*?\n',
            r'Thinking:.*?\n',
            r'Let me think.*?\n',
            r'<thinking>.*?</thinking>',
            r'\[system\].*?\[/system\]',
            r'<system>.*?</system>',
            r'System:.*?\n'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Clean up any extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
        
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
            
            # Get the raw response
            raw_response = response.json()["response"]
            
            # Clean the response by removing any thinking patterns
            cleaned_response = self.clean_response(raw_response)
            
            return cleaned_response
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Ollama API response: {str(e)}"

def find_relevant_chunks(query: str, chunks: List[str], embeddings: Dict, top_k: int = 3) -> List[str]:
    """Find most relevant document chunks for the query."""
    # Ensure the query is encoded using the same model
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
        prompt = f"""
You are a helpful assistant trained to provide answers based on the context provided. Below is the relevant context extracted from a document:

Context:
{context}

The user has asked the following question:

Query: {query}

Please respond by analyzing the context above and providing an answer that directly addresses the user's query. Make sure to use information from the provided context and provide a clear, concise, and direct response. Avoid unnecessary filler and be as helpful as possible.
"""
    else:
        # If no document context is available, use the query directly
        prompt = f"""
The user has asked the following question:

Query: {query}

Please provide a clear, concise, and direct response.
"""
    
    # Generate response
    return chatbot.generate_response(prompt)
