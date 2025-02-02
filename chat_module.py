# chat_module.py
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ChatBot:
    def __init__(self):
        # Initialize DeepSeek model and tokenizer
        self.model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_response(self, prompt: str) -> str:
        """Generate response using DeepSeek model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

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
        prompt = f"Query: {query}\n\nPlease provide a helpful response."
    
    # Generate response using the DeepSeek model
    try:
        response = chatbot.generate_response(prompt)
        
        # Clean up the response if needed
        # Remove any system prompts or prefixes that might be in the response
        if "Query:" in response:
            response = response.split("Query:")[0].strip()
        
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error while generating the response: {str(e)}"