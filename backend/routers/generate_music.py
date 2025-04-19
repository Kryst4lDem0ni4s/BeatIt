import os
import shutil
from fastapi import APIRouter, Depends, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uuid
from datetime import datetime
from auth import get_current_user
from ..training.lyricsgen import generate_lyrics
from ..training.vocalgen import generate_vocals
from ..training.instrumentalgen import InstrumentalGenerator
from ..training.musicgen import MusicGenerator
from config import storage_config

router = APIRouter()

# Request Models
class StyleReferences(BaseModel):
    lyrics_references: Optional[List[str]] = None
    vocals_references: Optional[List[str]] = None
    music_references: Optional[List[str]] = None

class MusicGenerationRequest(BaseModel):
    text_prompt: str
    vocal_settings: Optional[str] = "with vocals"  # "no vocals", "only vocals", "with vocals"
    vocals_source: Optional[str] = "generate vocals"  # "custom input", "generate vocals"
    instrumental_source: Optional[str] = "generate track"  # "custom input", "generate track"
    lyrics_settings: Optional[str] = "generate lyrics"  # "no lyrics", "custom lyrics", "generate lyrics"
    custom_lyrics: Optional[str] = None
    style_references: Optional[StyleReferences] = None
    reference_usage_mode: Optional[str] = "guidance"  # "guidance", "direct usage"
    pitch: Optional[int] = None
    tempo: Optional[int] = None
    styles_themes: Optional[List[str]] = []
    styles_themes_to_avoid: Optional[List[str]] = []
    instruments: Optional[List[str]] = []
    instruments_to_avoid: Optional[List[str]] = []

class MusicGenerationResponse(BaseModel):
    status: str
    message: str
    track_id: str
    audio_url: str
    metadata: Dict[str, Any]

@router.post("/generate-music", response_model=MusicGenerationResponse)
async def generate_music(
    request: MusicGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["uid"]
        
        # Generate a unique track ID
        track_id = str(uuid.uuid4())
        
        # Process lyrics based on settings
        lyrics = None
        if request.lyrics_settings == "generate lyrics":
            lyrics = generate_lyrics(
                prompt=request.text_prompt,
                reference_lyrics=request.style_references.lyrics_references if request.style_references else None,
                reference_mode=request.reference_usage_mode
            )
        elif request.lyrics_settings == "custom lyrics" and request.custom_lyrics:
            lyrics = request.custom_lyrics
        
        # Process vocals based on settings
        vocal_track_path = None
        if request.vocal_settings in ["with vocals", "only vocals"]:
            if request.vocals_source == "generate vocals" and lyrics:
                vocal_track_path = generate_vocals(
                    lyrics=lyrics,
                    pitch=request.pitch,
                    reference_tracks=request.style_references.vocals_references if request.style_references else None,
                    reference_mode=request.reference_usage_mode,
                    user_id=user_id
                )
            # Handle custom vocals input case (would need file upload handling)
        
        # Process instrumental based on settings
        instrumental_track_path = None
        if request.vocal_settings in ["with vocals", "no vocals"]:
            if request.instrumental_source == "generate track":
                instrumental_track_path = InstrumentalGenerator.create_midi(
                    prompt=request.text_prompt,
                    tempo=request.tempo,
                    desired_instruments=request.instruments,
                    avoided_instruments=request.instruments_to_avoid,
                    styles=request.styles_themes,
                    avoided_styles=request.styles_themes_to_avoid,
                    reference_tracks=request.style_references.music_references if request.style_references else None,
                    reference_mode=request.reference_usage_mode,
                    user_id=user_id
                )
            # Handle custom instrumental input case (would need file upload handling)
        
        # Combine tracks as needed
        final_track_path = None
        if vocal_track_path and instrumental_track_path:
            final_track_path = MusicGenerator.combine_tracks(
                vocals_path=vocal_track_path,
                instrumental_path=instrumental_track_path,
                user_id=user_id,
                track_id=track_id
            )
        elif vocal_track_path:
            final_track_path = vocal_track_path
        elif instrumental_track_path:
            final_track_path = instrumental_track_path
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio could be generated with the provided settings"
            )
        
        # Generate public URL for the track
        storage_path = f"users/{user_id}/tracks/{track_id}/final.mp3"
        audio_url = storage_config.upload_file(
            local_path=final_track_path,
            storage_path=storage_path
        )
        
        # Build metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,
            "prompt": request.text_prompt,
            "lyrics_mode": request.lyrics_settings,
            "vocal_mode": request.vocal_settings,
            "has_lyrics": request.lyrics_settings != "no lyrics",
            "has_vocals": request.vocal_settings != "no vocals",
            "has_instrumental": request.vocal_settings != "only vocals",
            "tempo": request.tempo,
            "pitch": request.pitch,
            "styles_themes": request.styles_themes,
            "instruments": request.instruments
        }
        
        # Store metadata in database
        # db.tracks.insert_one({...}) # Uncomment and implement based on your DB
        
        return MusicGenerationResponse(
            status="success",
            message="Music generated successfully",
            track_id=track_id,
            audio_url=audio_url,
            metadata=metadata
        )
    
    except Exception as e:
        # Log the error
        print(f"Error in music generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate music: {str(e)}"
        )

# Define valid audio file types and file size limits
VALID_AUDIO_TYPES = ["vocals", "instrumental", "reference"]
VALID_AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Response model
class AudioUploadResponse(BaseModel):
    status: str
    file_id: str
    file_url: str

@router.post("/upload-audio", response_model=AudioUploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    type: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload an audio file for use in music generation.
    
    Parameters:
    - file: Audio file (mp3, wav, ogg, m4a, flac)
    - type: Type of audio (vocals, instrumental, reference)
    - description: Optional context about the file
    
    Returns:
    - status: Success or failure
    - file_id: Unique identifier for the file
    - file_url: URL to access the file
    """
    # Get user ID from authenticated user
    user_id = current_user["uid"]
    
    # Validate file type
    if type not in VALID_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio type. Must be one of: {', '.join(VALID_AUDIO_TYPES)}"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in VALID_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Supported formats: {', '.join(VALID_AUDIO_EXTENSIONS)}"
        )
    
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Create temporary file to save upload
        temp_dir = f"temp/{user_id}/{file_id}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"original{file_extension}")
        
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            # Read file in chunks to handle large files efficiently
            content = await file.read(1024 * 1024)  # Read 1MB at a time
            file_size = 0
            
            while content:
                file_size += len(content)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB"
                    )
                buffer.write(content)
                content = await file.read(1024 * 1024)
        
        # Define storage path based on file type and user
        storage_path = f"users/{user_id}/{type}/{file_id}{file_extension}"
        
        # Upload file to configured storage
        file_url = storage_config.upload_file(
            local_path=temp_file_path,
            storage_path=storage_path
        )
        
        # Store metadata (could be expanded to use a database)
        metadata = {
            "user_id": user_id,
            "file_id": file_id,
            "original_filename": file.filename,
            "description": description,
            "type": type,
            "size": file_size,
            "extension": file_extension,
            "uploaded_at": datetime.now().isoformat(),
            "storage_path": storage_path
        }
        
        # Optional: Save metadata to database
        # db.audio_files.insert_one(metadata)
        
        # Clean up temporary file
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return AudioUploadResponse(
            status="success",
            file_id=file_id,
            file_url=file_url
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        print(f"Error in file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
        
GENERATION_STATUSES = {
    "QUEUED": "queued",
    "PROCESSING": "processing",
    "COMPLETED": "completed",
    "FAILED": "failed"
}

# Response model
class GenerationStatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    estimated_time: Optional[int] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@router.get("/generation-status/{track_id}", response_model=GenerationStatusResponse, 
           status_code=status.HTTP_200_OK)
async def get_generation_status(
    track_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Check the status of a music generation task.
    
    Parameters:
    - track_id: The ID of the track being generated
    
    Returns:
    - status: Current status (queued, processing, completed, failed)
    - progress: Percentage complete (if available)
    - estimated_time: Estimated time remaining in seconds (if available)
    - message: Additional status information
    - details: Additional details about the generation process
    """
    try:
        # Get user ID from authenticated user
        user_id = current_user["uid"]
        
        # Check if track exists and belongs to the user
        # You would typically query your database here
        # For demonstration, this is simulated
        track_info = get_track_info(track_id, user_id)
        
        if not track_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Track with ID {track_id} not found or does not belong to you"
            )
        
        # Get generation status from the task management system
        generation_status = get_generation_status_from_task_manager(track_id)
        
        return generation_status
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        print(f"Error fetching generation status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch generation status: {str(e)}"
        )

def get_track_info(track_id: str, user_id: str) -> Dict[str, Any]:
    """
    Fetch track information from the database.
    In a real implementation, this would query your database.
    
    This function should be replaced with actual database queries.
    """
    # Simulated track info - replace with database query
    # In production, you would check your database to verify
    # that this track exists and belongs to the user
    
    # For demonstration purposes only
    return {
        "track_id": track_id,
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "prompt": "Upbeat summer dance track"
    }

def get_generation_status_from_task_manager(track_id: str) -> GenerationStatusResponse:
    """
    Get the generation status from your task manager or database.
    
    In a real implementation, this would query a task queue like Celery
    or a database table tracking task progress.
    
    This function should be replaced with actual task status queries.
    """
    # This is a placeholder implementation
    # In production, you would check a task queue or database
    
    # Simple simulation based on track_id to show different statuses
    # In production, replace with real status checks
    track_id_sum = sum(ord(c) for c in track_id)
    status_index = track_id_sum % 4
    
    if status_index == 0:
        return GenerationStatusResponse(
            status=GENERATION_STATUSES["QUEUED"],
            message="Your music generation request is in the queue and will be processed soon",
            estimated_time=120  # 2 minutes
        )
    elif status_index == 1:
        return GenerationStatusResponse(
            status=GENERATION_STATUSES["PROCESSING"],
            progress=45.0,
            estimated_time=60,  # 1 minute
            message="Processing your music request - working on vocals",
            details={
                "current_step": "vocal_synthesis",
                "completed_steps": ["lyrics_generation", "beat_selection"],
                "pending_steps": ["instrumental_mixing", "mastering"]
            }
        )
    elif status_index == 2:
        return GenerationStatusResponse(
            status=GENERATION_STATUSES["COMPLETED"],
            progress=100.0,
            estimated_time=0,
            message="Your music has been successfully generated",
            details={
                "download_url": f"/api/download/{track_id}",
                "completed_at": datetime.now().isoformat(),
                "duration": "3:45"
            }
        )
    else:
        return GenerationStatusResponse(
            status=GENERATION_STATUSES["FAILED"],
            message="Music generation failed due to an error",
            details={
                "error": "Model inference failed",
                "error_details": "The AI model encountered an error with your specific prompt, please try again with different parameters"
            }
        )
        
# Define the response models
class TrackSummary(BaseModel):
    track_id: str
    title: Optional[str] = None
    created_at: datetime
    status: str
    audio_url: Optional[str] = None
    duration: Optional[str] = None
    prompt: str

class TrackDetail(BaseModel):
    track_id: str
    title: Optional[str] = None
    created_at: datetime
    status: str
    audio_url: Optional[str] = None
    duration: Optional[str] = None
    prompt: str
    lyrics: Optional[str] = None
    vocal_settings: str
    instrumental_settings: dict
    styles_themes: List[str]
    instruments: List[str]
    metadata: Dict[str, Any]
    
class DeleteResponse(BaseModel):
    status: str
    message: str

@router.get("/user-tracks", response_model=List[TrackSummary])
async def get_user_tracks(
    current_user: dict = Depends(get_current_user)
):
    """
    Get a list of all tracks generated by the authenticated user.
    """
    try:
        user_id = current_user["uid"]
        
        # In a real implementation, fetch tracks from database
        # Here we're using a placeholder function
        tracks = get_user_tracks_from_db(user_id)
        
        return tracks
        
    except Exception as e:
        # Log the error
        print(f"Error fetching user tracks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user tracks: {str(e)}"
        )

@router.get("/user-tracks/{track_id}", response_model=TrackDetail)
async def get_track_detail(
    track_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific track.
    """
    try:
        user_id = current_user["uid"]
        
        # Fetch the specific track from database
        track = get_track_detail_from_db(track_id, user_id)
        
        if not track:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Track with ID {track_id} not found or does not belong to you"
            )
        
        return track
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching track details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch track details: {str(e)}"
        )

@router.delete("/user-tracks/{track_id}", response_model=DeleteResponse)
async def delete_track(
    track_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a specific track.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if track exists and belongs to user
        track = get_track_detail_from_db(track_id, user_id)
        
        if not track:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Track with ID {track_id} not found or does not belong to you"
            )
        
        # Delete the track from storage and database
        delete_track_from_storage_and_db(track_id, user_id)
        
        return DeleteResponse(
            status="success",
            message=f"Track with ID {track_id} has been successfully deleted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting track: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete track: {str(e)}"
        )

# Database interaction functions
def get_user_tracks_from_db(user_id: str) -> List[TrackSummary]:
    """
    Fetch user tracks from the database.
    Replace with actual database query implementation.
    """
    # This is a placeholder. In a real application, you would query your database
    # Example with MongoDB:
    # tracks = db.tracks.find({"user_id": user_id})
    # return [TrackSummary(**track) for track in tracks]
    
    # For demonstration purposes only
    return [
        TrackSummary(
            track_id="track1",
            title="Summer Pop Beat",
            created_at=datetime.now(),
            status="completed",
            audio_url=f"https://storage.example.com/users/{user_id}/tracks/track1.mp3",
            duration="3:45",
            prompt="A summery pop beat with energetic synths"
        ),
        TrackSummary(
            track_id="track2",
            title="Jazz Fusion",
            created_at=datetime.now(),
            status="processing",
            audio_url=None,
            duration=None,
            prompt="Jazz fusion track with saxophone and electric piano"
        )
    ]

def get_track_detail_from_db(track_id: str, user_id: str) -> Optional[TrackDetail]:
    """
    Fetch detailed track information from the database.
    Replace with actual database query implementation.
    """
    # This is a placeholder. In a real application, you would query your database
    # Example with MongoDB:
    # track = db.tracks.find_one({"track_id": track_id, "user_id": user_id})
    # if track:
    #     return TrackDetail(**track)
    # return None
    
    # For demonstration purposes only
    if track_id == "track1":
        return TrackDetail(
            track_id="track1",
            title="Summer Pop Beat",
            created_at=datetime.now(),
            status="completed",
            audio_url=f"https://storage.example.com/users/{user_id}/tracks/track1.mp3",
            duration="3:45",
            prompt="A summery pop beat with energetic synths",
            lyrics="Summer days, feeling the heat wave...",
            vocal_settings="with vocals",
            instrumental_settings={
                "tempo": 120,
                "key": "C Major"
            },
            styles_themes=["pop", "summer", "energetic"],
            instruments=["synth", "drums", "bass"],
            metadata={
                "created_at": datetime.now().isoformat(),
                "format": "mp3",
                "size_mb": 4.2
            }
        )
    return None

def delete_track_from_storage_and_db(track_id: str, user_id: str):
    """
    Delete track from storage and database.
    Replace with actual implementation.
    """
    # Delete the audio file from your storage system
    storage_path = f"users/{user_id}/tracks/{track_id}/final.mp3"
    storage_config.delete_file(storage_path)
    
    # Delete the track record from your database
    # Example with MongoDB:
    # db.tracks.delete_one({"track_id": track_id, "user_id": user_id})
    
# Define valid file types and extensions
VALID_FILE_TYPES = ["vocals", "instrumental", "reference", "lyrics"]
VALID_FILE_PURPOSES = ["direct_use", "inspiration"]
VALID_AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
VALID_LYRICS_EXTENSIONS = [".txt", ".md"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Response models
class FileUploadResponse(BaseModel):
    file_id: str
    file_url: str
    status: str

class FileMetadata(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    purpose: str
    size: int
    upload_date: datetime
    file_url: str
    user_id: str
    content_type: str
    duration: Optional[float] = None

class FileDeleteResponse(BaseModel):
    status: str
    message: str

# 1.1 File Upload Endpoint
@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_type: str = Form(...),
    purpose: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload an audio or lyrics file for music generation.
    
    Parameters:
    - file: The file to upload
    - file_type: Type of file (vocals, instrumental, reference, lyrics)
    - purpose: How the file will be used (direct_use, inspiration)
    
    Returns:
    - file_id: Unique identifier for the uploaded file
    - file_url: URL to access the uploaded file
    - status: Success/failure message
    """
    try:
        # Get user ID from authenticated user
        user_id = current_user["uid"]
        
        # Validate file type
        if file_type not in VALID_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Must be one of: {', '.join(VALID_FILE_TYPES)}"
            )
            
        # Validate purpose
        if purpose not in VALID_FILE_PURPOSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid purpose. Must be one of: {', '.join(VALID_FILE_PURPOSES)}"
            )
        
        # Get file extension and validate based on file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_type == "lyrics" and file_extension not in VALID_LYRICS_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid lyrics file format. Supported formats: {', '.join(VALID_LYRICS_EXTENSIONS)}"
            )
        elif file_type != "lyrics" and file_extension not in VALID_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio file format. Supported formats: {', '.join(VALID_AUDIO_EXTENSIONS)}"
            )
        
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Create temporary file to save upload
        temp_dir = f"temp/{user_id}/{file_id}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"original{file_extension}")
        
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            # Read file in chunks to handle large files efficiently
            content = await file.read(1024 * 1024)  # Read 1MB at a time
            file_size = 0
            
            while content:
                file_size += len(content)
                if file_size > MAX_FILE_SIZE:
                    # Clean up the temp dir if file is too large
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB"
                    )
                buffer.write(content)
                content = await file.read(1024 * 1024)
        
        # Define storage path based on file type and user
        storage_path = f"users/{user_id}/{file_type}/{file_id}{file_extension}"
        
        # Upload file to configured storage
        file_url = storage_config.upload_file(
            local_path=temp_file_path,
            storage_path=storage_path
        )
        
        # Store metadata in database
        metadata = {
            "file_id": file_id,
            "file_name": file.filename,
            "file_type": file_type,
            "purpose": purpose,
            "size": file_size,
            "upload_date": datetime.now(),
            "file_url": file_url,
            "user_id": user_id,
            "content_type": file.content_type,
            "storage_path": storage_path
        }
        
        # In a real implementation, save this metadata to your database
        # db.files.insert_one(metadata)
        
        # Schedule temporary directory cleanup as background task
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        
        return FileUploadResponse(
            file_id=file_id,
            file_url=file_url,
            status="success"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

# 1.2 File Retrieval Endpoint
@router.get("/files/{file_id}", response_model=FileMetadata)
async def get_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve metadata about an uploaded file.
    
    Parameters:
    - file_id: Unique identifier for the file
    
    Returns:
    - File metadata including URL to access the file
    """
    try:
        user_id = current_user["uid"]
        
        # In a real implementation, fetch file metadata from your database
        # metadata = db.files.find_one({"file_id": file_id, "user_id": user_id})
        
        # For demonstration, we'll simulate fetching metadata
        metadata = get_file_metadata_from_db(file_id, user_id)
        
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found or does not belong to you"
            )
        
        return FileMetadata(**metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving file metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve file metadata: {str(e)}"
        )

# 1.3 File Deletion Endpoint
@router.delete("/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete an uploaded file.
    
    Parameters:
    - file_id: Unique identifier for the file
    
    Returns:
    - Confirmation of deletion
    """
    try:
        user_id = current_user["uid"]
        
        # Fetch file metadata from database to confirm ownership and get storage path
        # metadata = db.files.find_one({"file_id": file_id, "user_id": user_id})
        
        # For demonstration, we'll simulate fetching metadata
        metadata = get_file_metadata_from_db(file_id, user_id)
        
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found or does not belong to you"
            )
        
        # Delete the file from storage
        storage_config.delete_file(metadata["storage_path"])
        
        # Delete the file metadata from database
        # db.files.delete_one({"file_id": file_id, "user_id": user_id})
        
        return FileDeleteResponse(
            status="success",
            message=f"File with ID {file_id} has been successfully deleted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )

# Helper function (replace with database implementation)
def get_file_metadata_from_db(file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Placeholder function to simulate fetching file metadata from database.
    Replace with actual database query in production.
    """
    # This is for demonstration only
    if file_id == "sample-file-id":
        return {
            "file_id": file_id,
            "file_name": "sample_vocals.mp3",
            "file_type": "vocals",
            "purpose": "direct_use",
            "size": 4500000,
            "upload_date": datetime.now(),
            "file_url": f"https://storage.example.com/users/{user_id}/vocals/{file_id}.mp3",
            "user_id": user_id,
            "content_type": "audio/mpeg",
            "storage_path": f"users/{user_id}/vocals/{file_id}.mp3",
            "duration": 180.5
        }
    return None