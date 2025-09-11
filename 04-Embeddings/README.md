# 🧮 What are Embeddings?

## Overview
Embeddings are numerical representations of text, images, or other data that capture semantic meaning in high-dimensional vector space. They enable machines to understand and work with human language and other complex data types.

## 🎯 Learning Objectives
- Understand what embeddings are and how they work
- Learn about different types of embedding models
- Implement text embedding generation
- Use embeddings for similarity search and clustering

## 📚 Key Concepts

### 1. What are Embeddings?
Embeddings are dense vector representations that convert discrete objects (words, sentences, documents) into continuous numerical vectors that preserve semantic relationships.

### 2. Why Embeddings Matter
- **Semantic Understanding**: Capture meaning, not just syntax
- **Similarity Measurement**: Enable similarity calculations
- **Machine Learning**: Provide numerical input for ML models
- **Dimensionality**: Reduce complex data to manageable vectors

### 3. Key Properties
- **Dimensionality**: Typically 100-1000+ dimensions
- **Dense**: Most values are non-zero
- **Continuous**: Smooth vector space
- **Semantic**: Similar concepts have similar vectors

## 🔧 Types of Embeddings

### 1. Word Embeddings
Represent individual words as vectors.

#### Word2Vec
```python
from gensim.models import Word2Vec
import numpy as np

# Training data
sentences = [
    ['king', 'is', 'a', 'man'],
    ['queen', 'is', 'a', 'woman'],
    ['prince', 'is', 'a', 'boy']
]

# Train model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Get word embeddings
king_vector = model.wv['king']
queen_vector = model.wv['queen']

# Find similar words
similar = model.wv.most_similar('king', topn=5)
```

#### GloVe (Global Vectors)
```python
import numpy as np

# Load pre-trained GloVe vectors
def load_glove_vectors(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Usage
embeddings = load_glove_vectors('glove.6B.100d.txt')
king_vector = embeddings['king']
```

### 2. Sentence Embeddings
Represent entire sentences or phrases as vectors.

#### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
sentences = [
    "The cat sat on the mat",
    "A feline was resting on the rug",
    "The weather is sunny today"
]

embeddings = model.encode(sentences)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)
```

#### Universal Sentence Encoder
```python
import tensorflow_hub as hub
import numpy as np

# Load model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Generate embeddings
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A fast animal leaps over a sleeping canine"
]

embeddings = model(sentences)
similarity = np.inner(embeddings[0], embeddings[1])
```

### 3. Document Embeddings
Represent entire documents as vectors.

#### Doc2Vec
```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Prepare documents
documents = [
    TaggedDocument(words=['this', 'is', 'document', 'one'], tags=['doc1']),
    TaggedDocument(words=['this', 'is', 'document', 'two'], tags=['doc2']),
    TaggedDocument(words=['another', 'document', 'here'], tags=['doc3'])
]

# Train model
model = Doc2Vec(documents, vector_size=100, window=5, min_count=1)

# Get document embeddings
doc1_vector = model.dv['doc1']
doc2_vector = model.dv['doc2']

# Find similar documents
similar_docs = model.dv.most_similar('doc1')
```

## 🎯 Embedding Applications

### 1. Semantic Search
```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Document collection
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing deals with human language",
    "Computer vision processes and analyzes visual information"
]

# Generate embeddings
doc_embeddings = model.encode(documents)

# Search function
def semantic_search(query, documents, embeddings, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx]
        })
    return results

# Example search
results = semantic_search("AI and neural networks", documents, doc_embeddings)
for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Document: {result['document']}\n")
```

### 2. Clustering
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate embeddings for clustering
texts = [
    "I love pizza and pasta",
    "Italian food is delicious",
    "The weather is sunny today",
    "It's a beautiful day outside",
    "Machine learning is fascinating",
    "AI technology is advancing rapidly"
]

embeddings = model.encode(texts)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Visualize clusters
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('Document Clustering with Embeddings')
plt.show()

# Print clusters
for i in range(3):
    print(f"Cluster {i}:")
    cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
    for text in cluster_texts:
        print(f"  - {text}")
    print()
```

### 3. Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare data
texts = [
    "I love this product, it's amazing!",
    "This is terrible, I hate it",
    "Great service, highly recommended",
    "Poor quality, very disappointed",
    "Excellent customer support",
    "Worst experience ever"
]

labels = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# Generate embeddings
embeddings = model.encode(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.3, random_state=42
)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
```

## 🛠️ Advanced Embedding Techniques

### 1. Fine-tuning Embeddings
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
train_examples = [
    InputExample(texts=['Query 1', 'Relevant Document 1'], label=1.0),
    InputExample(texts=['Query 2', 'Relevant Document 2'], label=1.0),
    InputExample(texts=['Query 1', 'Irrelevant Document'], label=0.0),
]

# Create data loader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100
)
```

### 2. Multi-modal Embeddings
```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process text and image
text = "a photo of a cat"
image = "path/to/cat/image.jpg"  # Load your image

inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    text_embeddings = outputs.text_embeds
    image_embeddings = outputs.image_embeds

# Calculate similarity
similarity = torch.cosine_similarity(text_embeddings, image_embeddings)
```

## 🎯 Exercises

1. **Embedding Generation**: Create embeddings for different text types
2. **Similarity Analysis**: Compare embeddings of related and unrelated texts
3. **Clustering Project**: Group documents by semantic similarity
4. **Search Implementation**: Build a semantic search system

## 📖 Additional Resources

- [Embedding Models Comparison](./embedding-models.md)
- [Fine-tuning Guide](./fine-tuning-embeddings.md)
- [Multi-modal Embeddings](./multimodal-embeddings.md)
- [Embedding Visualization](./embedding-visualization.md)

## 🚀 Next Steps

After mastering embeddings, proceed to [Topic 5: What is LangChain?](../05-LangChain/)

---

**Key Takeaway**: Embeddings are the bridge between human language and machine understanding, enabling sophisticated AI applications through numerical representations of meaning.
