import boto3
import pinecone
import spacy
from pdfminer.high_level import extract_text
from transformers import BertTokenizer, BertModel
from moviepy.editor import VideoFileClip
import torch
from pydub import AudioSegment
import speech_recognition as sr
import os
import tempfile
import rdflib
from rdflib import Graph, URIRef
from fastapi import FastAPI, UploadFile, File
import shutil
import numpy as np
from tqdm import tqdm

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")
index = pinecone.Index("multimodal-data-index")

# Initialize BERT model for text embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Initialize SpaCy for Named Entity Recognition (NER) for knowledge graph
nlp = spacy.load("en_core_web_sm")

# AWS S3 client
s3_client = boto3.client('s3')

# FastAPI instance
app = FastAPI()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using pdfminer"""
    return extract_text(pdf_file)

def transcribe_audio_to_text(audio_file):
    """Transcribe audio to text using SpeechRecognition library"""
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

def extract_video_text(video_file):
    """Extract audio from video and transcribe it"""
    video_clip = VideoFileClip(video_file)
    audio_path = "/tmp/temp_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    return transcribe_audio_to_text(audio_path)

def chunk_text(text, chunk_size=512):
    """Chunk text into smaller pieces for embedding"""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def embed_text(text_chunks):
    """Generate embeddings using BERT model"""
    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings

def extract_entities(text):
    """Extract named entities for Knowledge Graph"""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def store_embeddings_in_pinecone(embeddings, metadata):
    """Store the embeddings and metadata in Pinecone"""
    vectors = [(f"{metadata['source']}chunk{i}", emb.tolist(), metadata) for i, emb in enumerate(embeddings)]
    index.upsert(vectors)

def create_knowledge_graph(entities, metadata):
    """Create a Knowledge Graph (RDF) with extracted entities"""
    graph = Graph()
    source_uri = URIRef(metadata['source'])
    
    for entity in entities:
        entity_uri = URIRef(f"{source_uri}/{entity}")
        graph.add((source_uri, URIRef("hasEntity"), entity_uri))
        
    graph.serialize(destination=f"/tmp/{metadata['source']}_kg.ttl", format="turtle")
    return graph

def process_pdf_from_s3(bucket_name, pdf_key):
    """Download PDF from S3, process it, and store embeddings in Pinecone and Knowledge Graph"""
    # Download the PDF file from S3
    temp_pdf_path = "/tmp/temp_pdf.pdf"
    s3_client.download_file(bucket_name, pdf_key, temp_pdf_path)
    
    # Extract text from the PDF
    text = extract_text_from_pdf(temp_pdf_path)
    
    # Chunk the text
    text_chunks = chunk_text(text)
    
    # Generate embeddings for the chunks
    embeddings = embed_text(text_chunks)
    
    # Extract entities for the Knowledge Graph
    entities = extract_entities(text)
    
    # Prepare metadata
    metadata = {
        "source": pdf_key,
        "num_chunks": len(text_chunks),
        "bucket_name": bucket_name
    }
    
    # Store the embeddings in Pinecone
    store_embeddings_in_pinecone(embeddings, metadata)
    
    # Create Knowledge Graph and save
    knowledge_graph = create_knowledge_graph(entities, metadata)
    
    # Clean up the temporary PDF file
    os.remove(temp_pdf_path)
    
    return knowledge_graph

@app.post("/process_data/")
async def process_data(bucket_name: str, file_key: str, data_type: str):
    """Trigger the processing of data from S3 and store in Pinecone and Knowledge Graph"""
    try:
        if data_type == "pdf":
            knowledge_graph = process_pdf_from_s3(bucket_name, file_key)
            return {"status": "success", "message": f"PDF {file_key} processed successfully.", "knowledge_graph": str(knowledge_graph)}
        elif data_type == "audio":
            # Download audio and process
            temp_audio_path = "/tmp/temp_audio.wav"
            s3_client.download_file(bucket_name, file_key, temp_audio_path)
            text = transcribe_audio_to_text(temp_audio_path)
            text_chunks = chunk_text(text)
            embeddings = embed_text(text_chunks)
            metadata = {"source": file_key, "num_chunks": len(text_chunks), "bucket_name": bucket_name}
            store_embeddings_in_pinecone(embeddings, metadata)
            os.remove(temp_audio_path)
            return {"status": "success", "message": f"Audio {file_key} processed successfully."}
        elif data_type == "video":
            # Download video and process
            temp_video_path = "/tmp/temp_video.mp4"
            s3_client.download_file(bucket_name, file_key, temp_video_path)
            text = extract_video_text(temp_video_path)
            text_chunks = chunk_text(text)
            embeddings = embed_text(text_chunks)
            metadata = {"source": file_key, "num_chunks": len(text_chunks), "bucket_name": bucket_name}
            store_embeddings_in_pinecone(embeddings, metadata)
            os.remove(temp_video_path)
            return {"status": "success", "message": f"Video {file_key} processed successfully."}
        else:
            return {"status": "error", "message": "Invalid data type"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)