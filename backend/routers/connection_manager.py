
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, status
from backend.models import model_types

class ConnectionManager:
    def __init__(self):
        # Store active connections by job_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store connection to user mapping for authentication
        self.connection_user_map: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str, user_id: str):
        await websocket.accept()
        
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        
        self.active_connections[job_id].append(websocket)
        self.connection_user_map[websocket] = user_id
        
        await websocket.send_json({
            "event": "connected",
            "job_id": job_id,
            "message": "Connected to generation progress updates"
        })
    
    # async def process_music_generation_job(job_id: str, user_id: str, parameters: dict):
    #     try:
    #         # Update job status to processing and send WebSocket update
    #         await update_job_status(job_id, "processing", 5, "Starting generation")
    #         await send_generation_update(job_id, "processing", 5, "Starting generation")
            
    #         # Process lyrics
    #         await update_job_status(job_id, "processing", 20, "Generating lyrics")
    #         await send_generation_update(job_id, "processing", 20, "Generating lyrics")
    #         lyrics = generate_lyrics(parameters["prompt"])
            
    #         # Process vocals
    #         await update_job_status(job_id, "processing", 40, "Generating vocals")
    #         await send_generation_update(job_id, "processing", 40, "Generating vocals")
    #         vocal_track = generate_vocals(lyrics)
            
    #         # Process instrumental
    #         await update_job_status(job_id, "processing", 70, "Generating instrumental")
    #         await send_generation_update(job_id, "processing", 70, "Generating instrumental")
    #         instrumental_track = generate_instrumental(parameters)
            
    #         # Final processing
    #         await update_job_status(job_id, "processing", 90, "Finalizing track")
    #         await send_generation_update(job_id, "processing", 90, "Finalizing track")
    #         final_track = combine_tracks(vocal_track, instrumental_track)
            
    #         # Complete
    #         await update_job_status(job_id, "completed", 100, "Generation complete")
    #         await send_generation_update(job_id, "completed", 100, "Generation complete")
            
    #     except Exception as e:
    #         await update_job_status(job_id, "failed", 0, f"Error: {str(e)}")
    #         await send_generation_update(job_id, "failed", 0, f"Error: {str(e)}")

    
    async def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections and websocket in self.active_connections[job_id]:
            self.active_connections[job_id].remove(websocket)
            
            if len(self.active_connections[job_id]) == 0:
                del self.active_connections[job_id]
        
        if websocket in self.connection_user_map:
            del self.connection_user_map[websocket]
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict, job_id: str):
        if job_id in self.active_connections:
            disconnected_websockets = []
            
            for websocket in self.active_connections[job_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected_websockets.append(websocket)
            
            for websocket in disconnected_websockets:
                await self.disconnect(websocket, job_id)

# Create a connection manager instance
connection_manager = ConnectionManager()

async def get_token_from_query(query_string: str) -> Optional[str]:
    """Extract token from query string"""
    params = query_string.split('&')
    for param in params:
        if param.startswith('token='):
            return param[6:]  # Remove 'token=' prefix
    return None

# Create WebSocket router
ws_router = APIRouter(tags=["WebSockets"])

@ws_router.websocket("/generation/{job_id}")
async def websocket_generation_progress(
    websocket: WebSocket,
    job_id: str
):
    """
    WebSocket endpoint for real-time generation progress updates.
    
    This endpoint provides updates on:
    - Progress percentage
    - Status changes
    - Completion notification
    """
    try:
        # Extract token from query parameters
        token = await get_token_from_query(websocket.query_string.decode())
        
        if not token:
            await websocket.close(code=1008, reason="Missing authentication token")
            return
        
        try:
            # Verify token and get user info
            from firebase_admin import auth
            decoded_token = auth.verify_id_token(token)
            user_id = decoded_token["uid"]
        except Exception as e:
            await websocket.close(code=1008, reason="Invalid authentication token")
            return
        
        # Check if job exists and belongs to user
        from ..models import Job
        job = await Job.get_by_id(job_id)
        
        if not job:
            await websocket.close(code=1003, reason="Job not found")
            return
            
        if job.user_id != user_id:
            await websocket.close(code=1003, reason="You don't have permission to access this job")
            return
        
        # Accept the connection
        await connection_manager.connect(websocket, job_id, user_id)
        
        # Send initial job status
        await connection_manager.send_personal_message(
            {
                "event": "status_update",
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "timestamp": datetime.now().isoformat()
            },
            websocket
        )
        
        # Keep connection open until client disconnects
        try:
            while True:
                data = await websocket.receive_text()
                # Acknowledge client message
                await connection_manager.send_personal_message(
                    {
                        "event": "acknowledged",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )
        except model_types.WebSocketDisconnect:
            await connection_manager.disconnect(websocket, job_id)
    
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Server error")
        except:
            pass

# Utility function to be called from background tasks to send updates
async def send_generation_update(job_id: str, status: str, progress: float, message: str):
    """
    Send a real-time update about generation progress.
    
    This function should be called from background tasks processing the generation.
    """
    await connection_manager.broadcast(
        {
            "event": "progress_update",
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        },
        job_id
    )