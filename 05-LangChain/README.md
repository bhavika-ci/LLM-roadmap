# 🔗 What is LangChain?

## Overview
LangChain is a powerful framework for building applications with Large Language Models (LLMs). It provides a unified interface for connecting LLMs to various data sources and tools, making it easier to build sophisticated AI applications.

## 🎯 Learning Objectives
- Understand what LangChain is and why it's used
- Learn about LangChain's core components
- Build applications using LangChain
- Integrate LangChain with vector databases and tools

## 📚 Key Concepts

### 1. What is LangChain?
LangChain is an open-source framework that simplifies the development of applications using LLMs by providing:
- **Modular Components**: Reusable building blocks
- **Chain Abstractions**: Connect different components together
- **Memory Management**: Handle conversation history
- **Tool Integration**: Connect to external APIs and databases

### 2. Why Use LangChain?
- **Simplified Development**: Reduces boilerplate code
- **Modularity**: Mix and match components
- **Extensibility**: Easy to add custom functionality
- **Production Ready**: Built for real-world applications

### 3. Core Components
- **LLMs**: Language model interfaces
- **Prompts**: Template management
- **Chains**: Sequential processing
- **Agents**: Tool-using systems
- **Memory**: Conversation state
- **Document Loaders**: Data ingestion

## 🔧 LangChain Components

### 1. LLMs and Chat Models
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Basic LLM
llm = OpenAI(temperature=0.7)
response = llm("Tell me a joke")

# Chat Model
chat = ChatOpenAI(temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of France?")
]
response = chat(messages)
```

### 2. Prompts and Prompt Templates
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Simple prompt template
template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Write a {audience}-friendly explanation of {topic}"
)
prompt = template.format(topic="quantum computing", audience="beginner")

# Chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a {role}"),
    HumanMessagePromptTemplate.from_template("{question}")
])
messages = chat_template.format_messages(
    role="data scientist",
    question="How do neural networks work?"
)
```

### 3. Chains
```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Simple chain
prompt = PromptTemplate(
    input_variables=["product"],
    template="Write a marketing slogan for {product}"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("eco-friendly water bottle")

# Sequential chain
first_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a summary of {topic}"
)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Translate this to Spanish: {summary}"
)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

sequential_chain = SimpleSequentialChain(
    chains=[first_chain, second_chain],
    verbose=True
)
result = sequential_chain.run("artificial intelligence")
```

### 4. Memory
```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Buffer memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Conversation with memory
response1 = conversation.predict(input="Hi, I'm John")
response2 = conversation.predict(input="What's my name?")

# Summary memory
summary_memory = ConversationSummaryMemory(llm=llm)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)
```

### 5. Document Loaders
```python
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# PDF loader
pdf_loader = PyPDFLoader("document.pdf")
pdf_documents = pdf_loader.load()

# Web loader
web_loader = WebBaseLoader("https://example.com")
web_documents = web_loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)
```

## 🛠️ RAG Implementation with LangChain

### Complete RAG Pipeline
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query the system
query = "What is the main topic of the document?"
result = qa_chain.run(query)
print(result)
```

### Advanced RAG with Custom Prompts
```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Custom prompt template
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

# Create chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)
```

## 🤖 LangChain Agents

### Tool-Using Agent
```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching current information on the internet"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use agent
result = agent.run("What is the latest news about AI?")
```

### Custom Tool
```python
from langchain.tools import BaseTool
from typing import Optional

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for performing mathematical calculations"
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)

# Use custom tool
calculator = CalculatorTool()
tools = [calculator]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("What is 15 * 23 + 45?")
```

## 🔗 LangChain with Vector Databases

### Chroma Integration
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Add documents
vectorstore.add_documents(texts)

# Similarity search
docs = vectorstore.similarity_search("query text", k=3)

# Similarity search with scores
docs_with_scores = vectorstore.similarity_search_with_score("query text", k=3)
```

### Pinecone Integration
```python
from langchain.vectorstores import Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")
index = pinecone.Index("your-index")

# Create vector store
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Add documents
vectorstore.add_documents(texts)

# Search
docs = vectorstore.similarity_search("query", k=3)
```

## 🎯 Practical Applications

### 1. Document Q&A System
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Chat interface
def chat_with_documents(question):
    result = conversational_chain({"question": question})
    return result["answer"]

# Example usage
answer = chat_with_documents("What are the main benefits mentioned?")
```

### 2. Code Generation Assistant
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Code generation prompt
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Generate clean, well-commented code."),
    ("human", "Write a Python function that {task}")
])

code_chain = LLMChain(llm=chat, prompt=code_prompt)

# Generate code
code_result = code_chain.run(task="calculates the factorial of a number")
print(code_result)
```

## 🎯 Exercises

1. **Basic Chain**: Create a simple LLM chain with custom prompts
2. **RAG System**: Build a document Q&A system with LangChain
3. **Agent Development**: Create an agent with multiple tools
4. **Memory Integration**: Implement conversation memory in a chatbot

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Advanced Chains](./advanced-chains.md)
- [Custom Agents](./custom-agents.md)
- [Production Deployment](./production-deployment.md)

## 🚀 Next Steps

After mastering LangChain, proceed to [Topic 6: Docker Container Setup](../06-Docker-Container/)

---

**Key Takeaway**: LangChain simplifies LLM application development by providing modular components and abstractions that handle complex workflows like RAG, memory management, and tool integration.
