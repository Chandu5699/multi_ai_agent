This project demonstrates a complete system for creating a multimodal data retrieval service with FastAPI, Nuxt.js, Docker, Pinecone vector database, and LangChain framework. The system involves embedding and storing various types of data (text, audio, video) and using a graph-based structure to query and retrieve knowledge from the Pinecone vector database.

Additionally, we'll integrate LangChain to create a Retrieval-Augmented Generation (RAG) agent that queries the knowledge graph and generates relevant responses.

Table of Contents

Overview
Tech Stack
Directory Structure
Setup Instructions
FastAPI Backend
Nuxt.js Frontend
Pinecone Integration
LangChain Agentic RAG
Docker Setup
GitHub Actions Workflow
License
Overview

This project uses a FastAPI backend as the API layer for querying multimodal data embeddings (text, audio, video) stored in Pinecone. The Nuxt.js frontend allows users to interact with the backend via a UI.

Key Features:
FastAPI: RESTful API to expose endpoints for querying the database.
Nuxt.js: A frontend that interacts with the FastAPI backend to display data.
Pinecone: A vector database to store multimodal data and embeddings.
LangChain: A framework to enable agentic RAG (Retrieval-Augmented Generation) using a knowledge graph structure.
Tech Stack

Backend: Python, FastAPI, Pinecone, LangChain
Frontend: Nuxt.js (Vue.js)
Database: Pinecone (Vector Database)
Containerization: Docker
CI/CD: GitHub Actions
APIs: Pinecone API, FastAPI endpoints
Machine Learning: BERT for text embeddings
GitHub Actions Workflow

The GitHub Actions workflow automates the CI/CD process, ensuring that every push to the repository triggers the build and deployment process for both FastAPI and Nuxt.js.

The workflow is set up in .github/workflows/ci.yml, and it will:

Build and deploy the FastAPI Docker container.
Build and deploy the Nuxt.js app.
Optionally deploy to a cloud service (e.g., Vercel, AWS).
License

This project implements a multi-agent system using graph-based Retrieval-Augmented Generation (RAG), designed for inference on multimodal data. It leverages FastAPI to Serve the model and provides a robust, scalable architecture using Docker and CI/CD with GitHub Actions. The agents are designed to handle PDF, audio, and video data, providing flexible and efficient inference.

Features:
Multi-agent system: Inference is distributed across multiple agents (master and child agents).
Graph-based RAG: Uses prompt engineering and RAG to improve performance and scalability.
Multimodal data processing: Supports PDFs, audio, and video files for inference.
Scalable: Batch processing and epoch handling for large-scale production use cases.
Dockerized: Ready-to-deploy containerized FastAPI application.
CI/CD pipeline: Automated testing and deployment using GitHub Actions.
Integration with Nuxt.js: Front-end interface to interact with the API.
Table of Contents

Installation
Project Structure
Usage
API Endpoints
Running with Docker
CI/CD Pipeline
Edge Case Handling
Scalability and Batch Processing
Future Improvements


This project is licensed under the MIT License. See the LICENSE file for more details.