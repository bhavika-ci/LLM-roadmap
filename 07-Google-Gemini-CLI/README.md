# 🤖 Google Gemini CLI

## Overview
Google Gemini CLI is a command-line interface tool that allows developers to interact with Google's Gemini Large Language Models directly from the terminal. It provides a convenient way to test prompts, generate content, and integrate Gemini into scripts and workflows.

## 🎯 Learning Objectives
- Understand what Google Gemini CLI is and its capabilities
- Learn how to install and configure Gemini CLI
- Use Gemini CLI for various tasks
- Integrate Gemini CLI into development workflows
- Build applications using Gemini CLI

## 📚 Key Concepts

### 1. What is Google Gemini CLI?
Google Gemini CLI is a command-line tool that provides direct access to Google's Gemini AI models, allowing developers to:
- Generate text and code from the terminal
- Test prompts and model responses
- Integrate AI capabilities into scripts
- Prototype AI applications quickly

### 2. Why Use Gemini CLI?
- **Rapid Prototyping**: Quick testing of AI capabilities
- **Script Integration**: Easy integration into automation workflows
- **Development Efficiency**: No need for complex API setup
- **Cost Effective**: Pay-per-use pricing model
- **Local Development**: Work offline with cached responses

### 3. Key Features
- Multiple model access (Gemini Pro, Gemini Pro Vision)
- Batch processing capabilities
- Configuration management
- Output formatting options
- Integration with development tools

## 🔧 Installation and Setup

### 1. Prerequisites
```bash
# Ensure you have Python 3.8+ installed
python --version

# Ensure you have pip installed
pip --version
```

### 2. Install Gemini CLI
```bash
# Install using pip
pip install google-generativeai

# Or install specific version
pip install google-generativeai==0.3.2

# Verify installation
python -c "import google.generativeai as genai; print('Installation successful')"
```

### 3. Authentication Setup
```bash
# Set up API key
export GOOGLE_API_KEY="your-api-key-here"

# Or create .env file
echo "GOOGLE_API_KEY=your-api-key-here" > .env

# Verify API key
python -c "import os; print('API Key:', os.getenv('GOOGLE_API_KEY'))"
```

### 4. Get API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for use in your applications

## 🛠️ Basic Usage

### 1. Simple Text Generation
```python
import google.generativeai as genai
import os

# Configure API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize model
model = genai.GenerativeModel('gemini-pro')

# Generate text
response = model.generate_content("Write a short story about a robot learning to paint")
print(response.text)
```

### 2. Command Line Usage
```bash
# Basic text generation
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Explain quantum computing in simple terms')
print(response.text)
"
```

### 3. Interactive Chat
```python
import google.generativeai as genai
import os

# Configure API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Start chat session
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Interactive conversation
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        break
    
    response = chat.send_message(user_input)
    print(f"Gemini: {response.text}\n")
```

## 🎯 Advanced Features

### 1. Multi-turn Conversations
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Start conversation with context
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["I'm learning about machine learning"]
    },
    {
        "role": "model", 
        "parts": ["That's great! Machine learning is a fascinating field. What specific aspect would you like to explore?"]
    }
])

# Continue conversation
response = chat.send_message("Tell me about neural networks")
print(response.text)
```

### 2. Code Generation
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Generate Python code
prompt = """
Write a Python function that:
1. Takes a list of numbers as input
2. Returns the sum of all even numbers
3. Includes proper error handling
4. Has docstring documentation
"""

response = model.generate_content(prompt)
print(response.text)
```

### 3. Batch Processing
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Process multiple prompts
prompts = [
    "Explain photosynthesis",
    "What is the capital of Japan?",
    "Write a haiku about coding"
]

responses = []
for prompt in prompts:
    response = model.generate_content(prompt)
    responses.append({
        'prompt': prompt,
        'response': response.text
    })

# Display results
for item in responses:
    print(f"Prompt: {item['prompt']}")
    print(f"Response: {item['response']}\n")
```

## 🔗 Integration with Development Tools

### 1. VS Code Extension
```json
// .vscode/settings.json
{
    "gemini.apiKey": "your-api-key",
    "gemini.model": "gemini-pro"
}
```

### 2. Jupyter Notebook Integration
```python
# In Jupyter notebook
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Cell magic for easy usage
def gemini_query(prompt):
    response = model.generate_content(prompt)
    return response.text

# Usage in notebook
result = gemini_query("Explain the concept of recursion")
print(result)
```

### 3. Shell Script Integration
```bash
#!/bin/bash
# gemini-helper.sh

API_KEY="your-api-key"
MODEL="gemini-pro"

query_gemini() {
    python3 -c "
import google.generativeai as genai
import os
genai.configure(api_key='$API_KEY')
model = genai.GenerativeModel('$MODEL')
response = model.generate_content('$1')
print(response.text)
"
}

# Usage
query_gemini "What is the weather like today?"
```

## 🎯 Practical Applications

### 1. Documentation Generator
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def generate_documentation(code):
    prompt = f"""
    Generate comprehensive documentation for this Python function:
    
    {code}
    
    Include:
    - Function description
    - Parameters explanation
    - Return value description
    - Usage examples
    - Error handling notes
    """
    
    response = model.generate_content(prompt)
    return response.text

# Example usage
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

docs = generate_documentation(code)
print(docs)
```

### 2. Code Review Assistant
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def review_code(code):
    prompt = f"""
    Review this Python code and provide feedback on:
    1. Code quality and best practices
    2. Potential bugs or issues
    3. Performance improvements
    4. Security considerations
    5. Suggestions for refactoring
    
    Code:
    {code}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Example usage
code_to_review = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

review = review_code(code_to_review)
print(review)
```

### 3. Test Case Generator
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def generate_tests(function_code):
    prompt = f"""
    Generate comprehensive unit tests for this Python function:
    
    {function_code}
    
    Include:
    - Normal case tests
    - Edge case tests
    - Error handling tests
    - Use pytest framework
    """
    
    response = model.generate_content(prompt)
    return response.text

# Example usage
function_code = """
def divide_numbers(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""

tests = generate_tests(function_code)
print(tests)
```

## 🚀 Building RAG Applications with Gemini CLI

### 1. Simple RAG Implementation
```python
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
import chromadb

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Initialize Chroma
client = chromadb.Client()
collection = client.create_collection("documents")

# Initialize embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def add_documents(texts):
    """Add documents to the vector database"""
    embeddings = embedding_model.encode(texts)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

def rag_query(question):
    """Perform RAG query using Gemini"""
    # Retrieve relevant documents
    query_embedding = embedding_model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    
    # Create context
    context = "\n".join(results['documents'][0])
    
    # Generate response with Gemini
    prompt = f"""
    Based on the following context, answer the question:
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    response = model.generate_content(prompt)
    return response.text

# Example usage
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing deals with human language"
]

add_documents(documents)
answer = rag_query("What is machine learning?")
print(answer)
```

### 2. Advanced RAG with LangChain
```python
import google.generativeai as genai
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize components
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings)

# Add documents
documents = [
    "Python is a high-level programming language",
    "Machine learning is used in many applications",
    "Docker helps with containerization"
]

vectorstore.add_documents(documents)

# Create custom prompt template
prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer: Let me think step by step.
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=None,  # We'll use Gemini directly
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)

def gemini_rag_query(question):
    """Enhanced RAG query with Gemini"""
    # Get relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create enhanced prompt
    prompt = f"""
    Based on the following context, answer the question:
    
    Context:
    {context}
    
    Question: {question}
    
    Please provide a comprehensive answer based on the context provided.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Example usage
answer = gemini_rag_query("What is Python used for?")
print(answer)
```

## 🎯 Best Practices

### 1. API Key Management
- Store API keys in environment variables
- Use .env files for local development
- Never commit API keys to version control
- Rotate keys regularly

### 2. Error Handling
```python
import google.generativeai as genai
import os

def safe_gemini_query(prompt, max_retries=3):
    """Safely query Gemini with error handling"""
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return f"Error: Failed to generate response after {max_retries} attempts"
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Rate Limiting
```python
import time
import google.generativeai as genai
import os

class RateLimitedGemini:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_content(self, prompt):
        # Rate limiting logic
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            time.sleep(sleep_time)
        
        self.requests.append(now)
        response = self.model.generate_content(prompt)
        return response.text
```

## 🎯 Exercises

1. **Basic CLI Usage**: Set up Gemini CLI and test basic functionality
2. **Code Generation**: Use Gemini to generate Python functions and classes
3. **RAG Implementation**: Build a simple RAG system with Gemini CLI
4. **Integration Project**: Create a development tool that uses Gemini CLI

## 📖 Additional Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Python SDK Documentation](https://github.com/google/generative-ai-python)
- [Best Practices Guide](./gemini-best-practices.md)

## 🚀 Final Project

After mastering all topics, build a complete **RAG-powered application** that:
- Uses Chroma vector database for document storage
- Implements semantic search with embeddings
- Integrates Google Gemini CLI for response generation
- Is containerized with Docker
- Provides a user-friendly interface

## 🎓 Course Completion

Congratulations! You've completed the comprehensive LLM Development Roadmap. You now have the knowledge and skills to:
- Understand and work with LLM models
- Implement RAG systems with vector databases
- Use embeddings for semantic search
- Build applications with LangChain
- Containerize applications with Docker
- Integrate Google Gemini CLI into your projects

---

**Key Takeaway**: Google Gemini CLI provides a powerful and accessible way to integrate AI capabilities into your development workflow, enabling rapid prototyping and production-ready applications.
