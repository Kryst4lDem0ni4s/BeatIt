from pydantic import EmailStr
from pydantic_settings import BaseSettings
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Settings(BaseSettings):
    # GOOGLE_APPLICATION_CREDENTIALS: str
    CREDENTIALS_FILE: str
    # DATABASE_URL: str

    class Config:
        env_file = './.env'
        
# config.py
import os
from google.cloud import storage
import firebase_admin
from firebase_admin import storage as firebase_storage

class StorageConfig:
    def __init__(self):
        self.bucket_name = os.getenv("STORAGE_BUCKET")
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
    
    def upload_file(self, local_path, storage_path):
        """Upload a file and return a public URL"""
        blob = self.bucket.blob(storage_path)
        blob.upload_from_filename(local_path)
        # Make file publicly accessible
        blob.make_public()
        return blob.public_url
    
    def delete_file(self, storage_path):
        """Delete a file from storage"""
        blob = self.bucket.blob(storage_path)
        blob.delete()
        return True

storage_config = StorageConfig()


# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# cred_file = os.path.join(base_dir, "Backend\app\helpers\kisaandvaar-firebase-adminsdk-t83e9-17bb4ada8c.json")
# settings = Settings(base_dir, cred_file)
settings = Settings()
