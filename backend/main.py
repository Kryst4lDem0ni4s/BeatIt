import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials
import firebase_admin
import uvicorn
from config import settings
from routers import auth, generate_music, lyrics, tracks, vocals, jobs, files, connection_manager, instrumentals
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
app.include_router(generate_music.router, tags=["Generate_Music"], prefix="/api/generate-music")
app.include_router(lyrics.router, tags=["Lyrics"], prefix="/api/lyrics")
app.include_router(vocals.router, tags=["Vocals"], prefix="/api/vocals")
app.include_router(instrumentals.router, tags=["Instrumentals"], prefix="/api/instrumentals")
app.include_router(tracks.router, tags=["Tracks"], prefix="/api/tracks")
app.include_router(jobs.router, tags=["Jobs"], prefix="/api/jobs")
app.include_router(files.router, tags=["Files"], prefix="/api/files")
app.include_router(connection_manager.router, tags=["Connection_Manager"], prefix="/api/connection-manager")

@app.get("/health")
async def root():
    logger.info("API is working fine.")
    return {"message": "API is working fine."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="127.0.0.1", port=8000, log_level="info", reload=True
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