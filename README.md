# LangChain RAG Application

## Overview
This project demonstrates a complete system for creating a multimodal data retrieval service with **FastAPI, Nuxt.js, Docker, Pinecone vector database, and the LangChain framework**. The system involves embedding and storing various types of data (text, audio, video) and using a **graph-based structure** to query and retrieve knowledge from the **Pinecone vector database**.

Additionally, we integrate **LangChain** to create a **Retrieval-Augmented Generation (RAG) agent** that queries the knowledge graph and generates relevant responses.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [FastAPI Backend](#fastapi-backend)
- [Nuxt.js Frontend](#nuxtjs-frontend)
- [Pinecone Integration](#pinecone-integration)
- [LangChain Agentic RAG](#langchain-agentic-rag)
- [Docker Setup](#docker-setup)
- [GitHub Actions Workflow](#github-actions-workflow)
- [License](#license)

## Tech Stack

- **Backend:** Python, FastAPI, Pinecone, LangChain
- **Frontend:** Nuxt.js (Vue.js)
- **Database:** Pinecone (Vector Database)
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **APIs:** Pinecone API, FastAPI endpoints
- **Machine Learning:** BERT for text embeddings

## Key Features

- **FastAPI**: RESTful API to expose endpoints for querying the database.
- **Nuxt.js**: A frontend that interacts with the FastAPI backend to display data.
- **Pinecone**: A vector database to store multimodal data and embeddings.
- **LangChain**: A framework to enable agentic RAG (Retrieval-Augmented Generation) using a knowledge graph structure.
- **Graph-based RAG**: Uses **prompt engineering** and **retrieval-augmented generation (RAG)** for efficient knowledge retrieval.
- **Multimodal Data Processing**: Supports **PDFs, audio, and video files** for inference.
- **Scalability**: Batch processing and epoch handling for large-scale production use cases.
- **Dockerized**: Ready-to-deploy containerized FastAPI application.
- **CI/CD Pipeline**: Automated testing and deployment using GitHub Actions.
- **Integration with Nuxt.js**: Provides a user-friendly interface to interact with the API.

## GitHub Actions Workflow

The **GitHub Actions** workflow automates the **CI/CD process**, ensuring that every push to the repository triggers the build and deployment process for both **FastAPI** and **Nuxt.js**.

The workflow is set up in `.github/workflows/ci.yml`, and it will:

- **Build and deploy** the FastAPI Docker container.
- **Build and deploy** the Nuxt.js app.
- **Optionally deploy** to a cloud service (e.g., **Vercel, AWS**).

## License

This project is licensed under the **MIT License**. See the LICENSE file for more details.

