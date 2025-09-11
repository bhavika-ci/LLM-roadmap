# 🗄️ Vector Database & Types

## Overview
Vector databases are specialized databases designed to store, index, and search high-dimensional vector data efficiently. They are essential for modern AI applications that rely on semantic similarity search.

## 🎯 Learning Objectives
- Understand what vector databases are and why they're needed
- Learn about different types of vector databases
- Compare vector database solutions
- Implement vector search operations

## 📚 Key Concepts

### 1. What is a Vector Database?
A vector database is a type of database optimized for storing and querying high-dimensional vectors (embeddings) that represent the semantic meaning of data.

### 2. Why Vector Databases?
- **Semantic Search**: Find similar content based on meaning, not exact matches
- **Scalability**: Handle millions of vectors efficiently
- **Performance**: Fast similarity search with optimized algorithms
- **AI Integration**: Designed for modern AI workflows

### 3. Key Features
- **Similarity Search**: Find nearest neighbors in vector space
- **Indexing**: Efficient data structures for fast retrieval
- **Filtering**: Combine vector search with metadata filtering
- **Scalability**: Handle large-scale vector operations

## 🔧 Types of Vector Databases

### 1. Dedicated Vector Databases
Purpose-built databases designed specifically for vector operations.

#### Chroma
```python
import chromadb

# Initialize Chroma
client = chromadb.Client()
collection = client.create_collection("my_collection")

# Add vectors
collection.add(
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["Document 1", "Document 2"],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_embeddings=[[1.1, 2.1, 3.1]],
    n_results=2
)
```

#### Pinecone
```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key")
index = pinecone.Index("my-index")

# Upsert vectors
index.upsert([
    ("vec1", [1.0, 2.0, 3.0], {"text": "Document 1"}),
    ("vec2", [4.0, 5.0, 6.0], {"text": "Document 2"})
])

# Query
results = index.query(
    vector=[1.1, 2.1, 3.1],
    top_k=2
)
```

#### Weaviate
```python
import weaviate

# Initialize Weaviate
client = weaviate.Client("http://localhost:8080")

# Create schema
schema = {
    "class": "Document",
    "properties": [
        {"name": "text", "dataType": ["string"]},
        {"name": "vector", "dataType": ["number[]"]}
    ]
}
client.schema.create_class(schema)

# Add objects
client.data_object.create({
    "text": "Document content",
    "vector": [1.0, 2.0, 3.0]
}, "Document")
```

### 2. Vector Search Engines
Open-source libraries and engines for vector operations.

#### FAISS (Facebook AI Similarity Search)
```python
import faiss
import numpy as np

# Create index
dimension = 128
index = faiss.IndexFlatL2(dimension)

# Add vectors
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

# Search
query_vector = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query_vector, k=5)
```

#### Annoy (Approximate Nearest Neighbors Oh Yeah)
```python
from annoy import AnnoyIndex

# Create index
dimension = 128
index = AnnoyIndex(dimension, 'angular')

# Add vectors
for i, vector in enumerate(vectors):
    index.add_item(i, vector)

# Build index
index.build(10)  # 10 trees

# Search
results = index.get_nns_by_vector(query_vector, 5)
```

### 3. Hybrid Databases
Traditional databases with vector search capabilities.

#### PostgreSQL with pgvector
```sql
-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384)
);

-- Create vector index
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Insert data
INSERT INTO documents (content, embedding) 
VALUES ('Document content', '[1,2,3,4,5]');

-- Search
SELECT content, embedding <-> '[1,2,3,4,5]' as distance
FROM documents
ORDER BY embedding <-> '[1,2,3,4,5]'
LIMIT 5;
```

#### Elasticsearch with Dense Vector
```json
{
  "mappings": {
    "properties": {
      "content": {"type": "text"},
      "embedding": {
        "type": "dense_vector",
        "dims": 384
      }
    }
  }
}
```

## 📊 Comparison Matrix

| Database | Type | Open Source | Cloud | Performance | Ease of Use |
|----------|------|-------------|-------|-------------|-------------|
| Chroma | Dedicated | ✅ | ✅ | High | Easy |
| Pinecone | Dedicated | ❌ | ✅ | Very High | Easy |
| Weaviate | Dedicated | ✅ | ✅ | High | Medium |
| FAISS | Engine | ✅ | ❌ | Very High | Hard |
| Annoy | Engine | ✅ | ❌ | High | Medium |
| pgvector | Hybrid | ✅ | ✅ | Medium | Medium |

## 🎯 Choosing the Right Vector Database

### For Beginners
- **Chroma**: Easy setup, good documentation
- **Pinecone**: Managed service, no infrastructure

### For Production
- **Pinecone**: Scalability and reliability
- **Weaviate**: Rich features and flexibility

### For Custom Solutions
- **FAISS**: Maximum performance control
- **pgvector**: Integration with existing databases

## 🛠️ Practical Implementation

### Setting up Chroma for RAG
```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Document processing pipeline
def add_documents(texts, metadatas=None):
    embeddings = model.encode(texts)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas or [{}] * len(texts),
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

# Search pipeline
def search_documents(query, n_results=5):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results
```

## 🎯 Exercises

1. **Database Comparison**: Set up Chroma, FAISS, and pgvector
2. **Performance Testing**: Compare search speeds across databases
3. **Scalability Test**: Test with different dataset sizes
4. **Integration Project**: Build a RAG system with your chosen database

## 📖 Additional Resources

- [Chroma Deep Dive](./chroma-deep-dive.md)
- [FAISS Optimization](./faiss-optimization.md)
- [Database Migration Guide](./database-migration.md)
- [Performance Benchmarking](./performance-benchmarks.md)

## 🚀 Next Steps

After mastering vector databases, proceed to [Topic 4: What are Embeddings?](../04-Embeddings/)

---

**Key Takeaway**: Vector databases are the backbone of modern AI applications, enabling efficient semantic search and similarity operations at scale.
