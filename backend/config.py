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
    GOOGLE_APPLICATION_CREDENTIALS: str
    CREDENTIALS_FILE: str
    DATABASE_URL: str

    class Config:
        env_file = './.env'

# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# cred_file = os.path.join(base_dir, "Backend\app\helpers\kisaandvaar-firebase-adminsdk-t83e9-17bb4ada8c.json")
# settings = Settings(base_dir, cred_file)
settings = Settings()
