import boto3
import os

class DataIngestion:
    def _init_(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def download_data(self, file_key, local_path):
        self.s3.download_file(self.bucket_name, file_key, local_path)
        print(f"Data downloaded to {local_path}")