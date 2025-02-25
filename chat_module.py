from typing import List, Dict, Optional
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ChatBot:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "deepseek-r1"
        
    def clean_response(self, text: str) -> str:
        """Clean the response by removing thinking patterns."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
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
            
            raw_response = response.json()["response"]
            
            cleaned_response = self.clean_response(raw_response)
            
            return cleaned_response
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Ollama API response: {str(e)}"

def find_relevant_chunks(query: str, chunks: List[str], vectorizer: TfidfVectorizer, top_k: int = 3) -> List[str]:
    """Find most relevant document chunks for the query using cosine similarity."""
    # vectorize the query, document chunks
    query_vec = vectorizer.transform([query])
    chunk_vecs = vectorizer.transform(chunks)
    
    # cosine similarities
    similarities = cosine_similarity(query_vec, chunk_vecs).flatten()
    
    # top k relevant chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def get_chat_response(query: str, chunks: Optional[List[str]] = None, embeddings: Optional[Dict] = None) -> str:
    chatbot = ChatBot()

    # ensure only actual text is passed, not embeddings
    if chunks and embeddings:
        relevant_chunks = find_relevant_chunks(query, chunks, embeddings['vectorizer'])
        context = '\n'.join(relevant_chunks) if relevant_chunks else "No relevant document context available."
    else:
        context = "No document uploaded."

    print(f"📝 Context used in chatbot:\n{context[:500]}") 

    if not any(char.isalpha() for char in context): 
        print("🚨 ERROR: No readable text in context!")
        context = "Document content could not be extracted."

    # prompt template, adjust here
    prompt = f"""
# Core Identity and Capabilities
You are a helpful AI assistant that combines accurate knowledge with appropriate confidence. You should provide information you're confident about while being honest about uncertainties.

# Knowledge Confidence Framework

## 1. Core Historical Facts (MUST STATE WITH CONFIDENCE)
- Well-documented historical events
- Current and former heads of state
- Major political positions and appointments
- Significant historical figures
- Basic country information
- Established organizations
→ Answer these directly and confidently

Example responses:
"Tun Dr. Mahathir Mohamad was Malaysia's fourth and seventh Prime Minister, serving from 1981-2003 and 2018-2020. He is known as Malaysia's longest-serving prime minister during his first tenure."

## 2. Recent Events and Changes
- State with appropriate time context
- Include qualifier about potential changes
→ Answer with time context

Example:
"As of late 2022, Anwar Ibrahim became Malaysia's 10th Prime Minister."

## 3. Uncertain or Incomplete Knowledge
- Specific details you're unsure about
- Rapidly changing situations
- Complex or contested information
→ State what you do know, be clear about uncertainties

# Response Protocol

## For Well-Known Facts:
1. Provide information confidently
2. Include relevant dates and context
3. Don't add unnecessary uncertainty disclaimers

## For Mixed Confidence:
1. State what you know confidently first
2. Separately note what's less certain
3. Explain any limitations clearly

## For Uncertain Information:
1. State what you do know
2. Explain specific gaps in knowledge
3. Don't default to complete uncertainty

# Document Context Handling
When {context} is provided:
- Use context information first
- Supplement with well-known facts when appropriate
- Clearly indicate when mixing sources

# Query Processing
User Query: {query}
Available Context: {context}

# Response Guidelines

## DO:
- Answer confidently about well-documented facts
- Provide relevant historical context
- State specific dates and roles
- Include significant achievements
- Acknowledge partial knowledge when applicable

## DON'T:
- Default to "I don't know" for well-known information
- Add unnecessary uncertainty to established facts
- Refuse to answer without context
- Ask for clarification about well-known figures
- Hide behind vague language

## Response Structure

### For Well-Known Figures/Facts:
```
[Direct statement of key facts]
[Relevant dates and roles]
[Significant context or achievements]
```

### For Partial Knowledge:
```
[State what is known confidently]
[Explain specific aspects that are uncertain]
[Provide relevant context available]
```

# Special Instructions
1. Be confident about well-documented historical facts
2. Don't require context for basic historical knowledge
3. Use "I don't know" only for genuinely uncertain information
4. Balance accuracy with appropriate confidence
5. Provide context and background for important figures
"""

    return chatbot.generate_response(prompt)




