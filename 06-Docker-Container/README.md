# 🐳 Docker Container Setup

## Overview
Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers. It ensures consistent deployment across different environments and simplifies application management.

## 🎯 Learning Objectives
- Understand what Docker containers are and why they're useful
- Learn how to install and set up Docker
- Create Dockerfiles for applications
- Build and run Docker containers
- Deploy containerized applications

## 📚 Key Concepts

### 1. What is Docker?
Docker is a platform that uses containerization technology to package applications with all their dependencies into standardized units called containers.

### 2. Why Use Docker?
- **Consistency**: Same environment across development, testing, and production
- **Portability**: Run anywhere Docker is installed
- **Isolation**: Containers don't interfere with each other
- **Scalability**: Easy to scale applications horizontally
- **Resource Efficiency**: Lightweight compared to virtual machines

### 3. Key Components
- **Docker Engine**: Runtime that manages containers
- **Docker Image**: Template for creating containers
- **Docker Container**: Running instance of an image
- **Dockerfile**: Instructions for building images
- **Docker Compose**: Tool for multi-container applications

## 🔧 Installation and Setup

### 1. Install Docker Desktop
```bash
# Windows (using Chocolatey)
choco install docker-desktop

# macOS (using Homebrew)
brew install --cask docker

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. Verify Installation
```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Test Docker installation
docker run hello-world
```

### 3. Docker Desktop Configuration
- Start Docker Desktop application
- Configure resources (CPU, Memory)
- Enable Kubernetes (optional)
- Set up Docker Hub account

## 🛠️ Creating Docker Containers

### 1. Basic Dockerfile
```dockerfile
# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Run application
CMD ["python", "app.py"]
```

### 2. Multi-stage Dockerfile
```dockerfile
# Build stage
FROM node:16 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Production stage
FROM node:16-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### 3. Python Application Example
```dockerfile
# Python Flask application
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

## 🚀 Building and Running Containers

### 1. Build Docker Image
```bash
# Build image with tag
docker build -t my-app:latest .

# Build with specific Dockerfile
docker build -f Dockerfile.prod -t my-app:prod .

# Build with build arguments
docker build --build-arg VERSION=1.0.0 -t my-app:1.0.0 .
```

### 2. Run Container
```bash
# Run container
docker run -d --name my-container my-app:latest

# Run with port mapping
docker run -d -p 8000:8000 --name my-container my-app:latest

# Run with environment variables
docker run -d -e DATABASE_URL=postgresql://... my-app:latest

# Run with volume mounting
docker run -d -v /host/path:/container/path my-app:latest
```

### 3. Container Management
```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop my-container

# Start container
docker start my-container

# Remove container
docker rm my-container

# View container logs
docker logs my-container

# Execute command in running container
docker exec -it my-container bash
```

## 🔗 Docker Compose for Multi-Container Applications

### 1. Basic docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://db:5432/myapp
    depends_on:
      - db
    volumes:
      - .:/app

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 2. Advanced docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://db:5432/myapp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d myapp"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:6-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
```

### 3. Docker Compose Commands
```bash
# Start all services
docker-compose up -d

# Build and start
docker-compose up --build

# Stop all services
docker-compose down

# View logs
docker-compose logs -f web

# Scale services
docker-compose up --scale web=3

# Execute command in service
docker-compose exec web bash
```

## 🎯 LLM Application Containerization

### 1. LangChain Application Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Requirements.txt for LLM App
```txt
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.0.350
openai==1.3.7
chromadb==0.4.18
sentence-transformers==2.2.2
python-dotenv==1.0.0
pydantic==2.5.0
```

### 3. Docker Compose for RAG Application
```yaml
version: '3.8'

services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_PERSIST_DIRECTORY=/app/chroma_db
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
    depends_on:
      - chroma
    restart: unless-stopped

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    restart: unless-stopped

volumes:
  chroma_data:
```

## 🔧 Advanced Docker Features

### 1. Multi-architecture Builds
```bash
# Build for multiple architectures
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t my-app:latest .
```

### 2. Docker Secrets
```yaml
version: '3.8'

services:
  web:
    image: my-app:latest
    secrets:
      - db_password
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### 3. Docker Networks
```bash
# Create custom network
docker network create my-network

# Run containers on custom network
docker run -d --network my-network --name web my-app:latest
docker run -d --network my-network --name db postgres:13
```

## 🎯 Best Practices

### 1. Security
- Use non-root users
- Keep base images updated
- Scan images for vulnerabilities
- Use multi-stage builds
- Don't store secrets in images

### 2. Performance
- Use .dockerignore file
- Leverage Docker layer caching
- Use specific base image tags
- Minimize image size
- Use health checks

### 3. Development
- Use docker-compose for local development
- Mount source code as volumes
- Use development-specific Dockerfiles
- Implement hot reloading

## 🎯 Exercises

1. **Basic Container**: Create a simple Python web app container
2. **Multi-service App**: Build a full-stack application with Docker Compose
3. **LLM App Containerization**: Containerize a LangChain RAG application
4. **Production Setup**: Create production-ready Docker configuration

## 📖 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Best Practices](./docker-best-practices.md)
- [Production Deployment](./production-deployment.md)
- [Security Hardening](./docker-security.md)

## 🚀 Next Steps

After mastering Docker, proceed to [Topic 7: Google Gemini CLI](../07-Google-Gemini-CLI/)

---

**Key Takeaway**: Docker provides a standardized way to package and deploy applications, ensuring consistency across environments and simplifying the deployment of complex AI applications.
