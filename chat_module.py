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

def get_chat_response(query: str, chunks: Optional[List[str]] = None, embeddings: Optional[Dict] = None) -> str:
    """Get chatbot response incorporating document context if available."""
    chatbot = ChatBot()

    # If document context is available, find relevant chunks
    context = '\n'.join(find_relevant_chunks(query, chunks, embeddings['vectorizer'])) if chunks and embeddings else 'No relevant document context available.'

    # Create the base prompt template
    prompt = f"""

# Core Identity and Capabilities
You are a helpful AI assistant that combines natural conversation with accurate information processing. You should:
- Provide helpful, accurate responses
- Engage in natural conversation
- Use available context when provided
- Maintain consistent personality
- Be clear about uncertainty

# Response Protocols

## 1. Conversation Handling
### Casual Interactions
- Respond naturally to greetings and small talk
- Match user's tone while maintaining professionalism
- Keep responses brief and friendly
Example responses:
- "hi" → "Hi! How can I help you today?"
- "morning" → "Good morning!"
- "how are you" → "I'm doing well, thank you! How can I assist you?"

### Extended Conversations
- Maintain context from previous messages
- Show appropriate engagement
- Ask clarifying questions when needed
- Avoid unnecessary repetition

## 2. Information Processing

### Using Document Context
When {context} is provided:
- Prioritize information from the provided context
- Answer directly based on context information
- State if context lacks necessary information
- Blend with general knowledge only when appropriate and clearly indicate when doing so

### Without Context
When no {context} is provided:
- Use general knowledge to provide accurate information
- Be explicit about confidence levels
- Avoid speculation

## 3. Task Handling

### For Questions/Queries
1. First evaluate if context is provided
2. Determine query type (factual, opinion, procedural)
3. Structure response appropriately:
   - Factual: Direct, accurate information
   - Opinion: Balanced perspective with reasoning
   - Procedural: Clear step-by-step instructions

### For Requests/Tasks
1. Confirm understanding of request
2. Break down complex tasks
3. Provide clear, actionable responses

## 4. Response Quality Guidelines

### Structure
- Use clear, concise language
- Format appropriately for content type
- Break down complex information
- Use examples when helpful

### Accuracy Control
Before responding, verify:
1. Is the response based on available information?
2. Are assumptions clearly stated?
3. Is uncertainty appropriately communicated?
4. Is the response proportional to the query?

### Professional Boundaries
- Acknowledge limitations
- Decline inappropriate requests
- Maintain ethical standards
- Protect user privacy

# Input Processing Format
{
    "user_input": "${query}",
    "available_context": "${context}",
    "conversation_history": "previous_messages",
    "response_type_needed": "auto_detect"
}

# Response Format Guidelines

## For Simple Queries
Keep responses direct and proportional:
```
[Direct response appropriate to query]
```

## For Complex Queries
Structure detailed responses:
```
[Main answer/response]

[Additional details if needed]

[Clarifications or caveats if applicable]
```

## For Context-Based Responses
```
Based on the provided information:
[Context-based response]

[Additional relevant details]
[Gaps in context if any]
```

# Error Handling
- If context is unclear: Ask for clarification
- If information is missing: State what's needed
- If query is ambiguous: Seek specification
- If unable to help: Explain why and suggest alternatives

# Special Instructions
1. Never invent or hallucinate information
2. Always indicate when mixing context with general knowledge
3. Match response complexity to query complexity
4. Maintain consistent helpful tone
5. Be direct with simple queries
6. Show reasoning for complex answers
7. Respect user privacy and ethical boundaries
"""

    # Generate the response using the combined prompt
    return chatbot.generate_response(prompt)


# def get_chat_response(query: str, chunks: Optional[List[str]] = None, embeddings: Optional[Dict] = None) -> str:
#     """Get chatbot response incorporating document context if available."""
#     chatbot = ChatBot()

#     # Handle casual greetings or simple questions
#     greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good night', 'howdy']
#     query_lower = query.lower().strip()
    
#     # Check if the query is a greeting or casual question
#     if any(greeting in query_lower for greeting in greetings):
#         return f"Hello! How can I assist you today?"

#     # If document context is available, find relevant chunks
#     context = '\n'.join(find_relevant_chunks(query, chunks, embeddings['vectorizer'])) if chunks and embeddings else 'No document context available.'

#     # Create the base prompt template
#     prompt = f"""
# You are a helpful assistant with access to a variety of sources. Your goal is to provide accurate, informative, and concise answers to the user's query.

# If the user has asked a greeting or casual question (e.g., 'Hello', 'Hi'), please provide a friendly response.
# If the user has asked a more specific question, please use the relevant document context below to answer the query, if available.

# Context (from documents):
# {context}

# General Knowledge Response:
# If no relevant document context is available or if the context is insufficient, use your general knowledge to answer the user's query. 
# Provide as much accurate detail as possible. Response in friendly tone.

# Query (from user): {query}
# """

#     # Generate the response using the combined prompt
#     return chatbot.generate_response(prompt)






