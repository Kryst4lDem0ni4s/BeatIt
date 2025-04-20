import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials
import firebase_admin
import uvicorn
from config import settings
from routers import auth, generate_music
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
app.include_router(auth.router, tags=["Authentication"], prefix="/api/generate-music")

@app.get("/health")
async def root():
    logger.info("API is working fine.")
    return {"message": "API is working fine."}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="127.0.0.1", port=8000, log_level="info", reload=True
    )
    
# from fastapi import FastAPI
# from routers.options import router as options_router
# from routers.websockets import ws_router

# app = FastAPI()

# # Include the options router
# app.include_router(options_router)

# # Include the WebSocket router
# app.include_router(ws_router, prefix="/ws")
    
# In your music generation background task
# 