import boto3
import pinecone
import torch
from transformers import BertTokenizer, BertModel
from fastapi import FastAPI, Query
import numpy as np
from scipy.spatial.distance import cosine

# Initialize Pinecone for querying the vector database
pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")
index = pinecone.Index("multimodal-data-index")

# Initialize BERT model for text embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# FastAPI instance
app = FastAPI()

def embed_text(text: str):
    """Generate embeddings for the input text using the BERT model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def query_pinecone(query_vector, top_k=5):
    """Query Pinecone for the most similar vectors to the input query."""
    results = index.query(query_vector.tolist(), top_k=top_k)
    return results['matches']

@app.get("/query/")
async def query(
    text: str = Query(..., description="Text to query for similar embeddings in Pinecone"),
    top_k: int = Query(5, le=10, description="Number of top results to retrieve")
):
    """Query the Pinecone vector database for the most similar text embeddings."""
    try:
        # Embed the query text using BERT
        query_vector = embed_text(text)

        # Query Pinecone for the most similar embeddings
        results = query_pinecone(query_vector, top_k)
        
        # Return the results
        return {"status": "success", "matches": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/similarity/")
async def similarity(
    text1: str = Query(..., description="First text to compare"),
    text2: str = Query(..., description="Second text to compare")
):
    """Calculate cosine similarity between two input texts."""
    try:
        # Generate embeddings for both texts
        vector1 = embed_text(text1)
        vector2 = embed_text(text2)

        # Compute cosine similarity
        similarity_score = 1 - cosine(vector1, vector2)

        return {"status": "success", "similarity_score": similarity_score}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)