from typing import List, Dict, Optional
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ChatBot:
    def __init__(self):
        # Initialize Ollama API settings
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "deepseek-r1"
        
    def clean_response(self, text: str) -> str:
        """Clean the response by removing thinking patterns."""
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

def find_relevant_chunks(query: str, chunks: List[str], vectorizer: TfidfVectorizer, top_k: int = 3) -> List[str]:
    """Find most relevant document chunks for the query using cosine similarity."""
    # Vectorize the query and the document chunks
    query_vec = vectorizer.transform([query])
    chunk_vecs = vectorizer.transform(chunks)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vec, chunk_vecs).flatten()
    
    # Get the indices of the top k relevant chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def get_chat_response(query: str, chunks: Optional[List[str]] = None, vectorizer: Optional[TfidfVectorizer] = None) -> str:
    """Get chatbot response incorporating document context if available."""
    chatbot = ChatBot()

    # Handle casual greetings or simple questions
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good night', 'howdy']
    query_lower = query.lower().strip()
    
    # Check if the query is a greeting or casual question
    if any(greeting in query_lower for greeting in greetings):
        return f"Hello! How can I assist you today?"

    # If document context is available, find relevant chunks
    context = '\n'.join(find_relevant_chunks(query, chunks, vectorizer)) if chunks and vectorizer else 'No document context available.'

    # Create the base prompt template
    prompt = f"""
You are a helpful assistant with access to a variety of sources. Your goal is to provide accurate, informative, and concise answers to the user's query.

If the user has asked a greeting or casual question (e.g., 'Hello', 'Hi'), please provide a friendly response.
If the user has asked a more specific question, please use the relevant document context below to answer the query, if available.

Context (from documents):
{context}

General Knowledge Response:
If no relevant document context is available or if the context is insufficient, use your general knowledge to answer the user's query. 
Provide as much accurate detail as possible. Response in friendly tone.

Query (from user): {query}
"""

    # Generate the response using the combined prompt
    return chatbot.generate_response(prompt)





