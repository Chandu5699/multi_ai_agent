import boto3
import os

class DataIngestion:
    def _init_(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def download_data(self, file_key, local_path):
        self.s3.download_file(self.bucket_name, file_key, local_path)
        print(f"Data downloaded to {local_path}")
import json

def chunk_json_file(input_file, chunk_size=100):
    """Chunk a large JSON file into smaller pieces of 'chunk_size' records."""
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Assume the JSON file is a list of records
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Yielding each chunk
    for chunk in chunks:
        yield chunk

# Example usage
input_file = "large_data.json"  # Replace with your JSON file path
for chunk in chunk_json_file(input_file, chunk_size=100):
    # Process each chunk (for example, send to RAG retrieval model)
    print(chunk)  # Replace with processing logic