import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials
import firebase_admin
import uvicorn
from config import settings
from routers import auth
from log_base import setup_logger

logger = setup_logger(__name__)

app = FastAPI()
load_dotenv()

cred_file = os.getenv("CREDENTIALS_FILE")
database_url = os.getenv("DATABASE_URL")
cred = credentials.Certificate(cred_file)
firebase_admin.initialize_app(credential=cred, options={"databaseURL":database_url})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, tags=["Authentication"], prefix="/api/auth")

@app.get("/health")
async def root():
    logger.info("API is working fine.")
    return {"message": "API is working fine."}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="127.0.0.1", port=8000, log_level="info", reload=True
    )