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

# Request models
class StyleReferences(BaseModel):
    lyrics_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for lyrics references")
    vocals_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for vocal references")
    music_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for music references")

class MusicGenerationInitRequest(BaseModel):
    text_prompt: str = Field(..., description="Textual description of desired music")
    generation_type: GenTypeEnum = Field("instrumental_vocal", description="Type of generation to perform")
    vocal_settings: VocalSettingsEnum = Field("with_vocals", description="Options for vocals")
    vocals_source: VocalsSourceEnum = Field("generate_vocals", description="Source of vocals")
    instrumental_source: InstrumentalSourceEnum = Field("generate_track", description="Source of instrumental track")
    lyrics_settings: LyricsSettingsEnum = Field("generate_lyrics", description="Lyrics options")
    custom_lyrics: Optional[str] = Field(None, description="Text content when custom lyrics are selected")
    style_references: Optional[StyleReferences] = Field(None, description="Object containing reference files")
    reference_usage_mode: ReferenceUsageModeEnum = Field("guidance_only", description="Whether references are for guidance only or direct usage")
    pitch: Optional[str] = Field(None, description="Desired pitch settings (e.g., 'C#')")
    tempo: Optional[int] = Field(None, description="Desired tempo in BPM (e.g., 120)")
    styles_themes: List[str] = Field(default=[], description="Array of desired music styles/themes")
    styles_themes_to_avoid: List[str] = Field(default=[], description="Array of styles/themes to avoid")
    instruments: List[str] = Field(default=[], description="Array of desired instruments")
    instruments_to_avoid: List[str] = Field(default=[], description="Array of instruments to avoid")

class MusicGenerationInitResponse(BaseModel):
    job_id: str
    status: str
    estimated_time: int  # in seconds

# In-memory job storage (replace with a database in production)
jobs_db = {}

@router.post("/generate/init", response_model=MusicGenerationInitResponse)
async def init_music_generation(
    request: MusicGenerationInitRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Initialize a music generation job with all parameters.
    
    This endpoint starts the music generation process asynchronously and returns a job ID immediately.
    """
    try:
        user_id = current_user["uid"]
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Estimate processing time based on complexity
        # This is a simplified example - in production, you would use a more sophisticated estimation
        base_time = 60  # base time in seconds
        if request.generation_type == "instrumental_vocal":
            estimated_time = base_time * 2
        elif request.generation_type == "vocal_only":
            estimated_time = base_time
        else:
            estimated_time = base_time * 1.5
            
        # Additional time for reference processing
        if request.style_references:
            if request.style_references.lyrics_references:
                estimated_time += len(request.style_references.lyrics_references) * 5
            if request.style_references.vocals_references:
                estimated_time += len(request.style_references.vocals_references) * 10
            if request.style_references.music_references:
                estimated_time += len(request.style_references.music_references) * 10
        
        # Create job record
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "queued",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "estimated_completion_time": datetime.now() + timedelta(seconds=estimated_time),
            "progress": 0,
            "parameters": request.dict(),
            "result": None,
            "error": None
        }
        
        # Store job data (in a real app, this would go to a database)
        jobs_db[job_id] = job_data
        
        # Start the generation process in the background
        background_tasks.add_task(
            process_music_generation_job,
            job_id=job_id,
            user_id=user_id,
            parameters=request.dict()
        )
        
        return MusicGenerationInitResponse(
            job_id=job_id,
            status="queued",
            estimated_time=int(estimated_time)
        )
        
    except Exception as e:
        print(f"Error initializing music generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize music generation: {str(e)}"
        )

async def process_music_generation_job(job_id: str, user_id: str, parameters: dict):
    """
    Background task to process a music generation job.
    This function is called asynchronously after the API returns.
    """
    try:
        # Update job status to processing
        jobs_db[job_id]["status"] = "processing"
        jobs_db[job_id]["updated_at"] = datetime.now()
        jobs_db[job_id]["progress"] = 5
        
        # Extract parameters
        text_prompt = parameters["text_prompt"]
        generation_type = parameters["generation_type"]
        vocal_settings = parameters["vocal_settings"]
        vocals_source = parameters["vocals_source"]
        instrumental_source = parameters["instrumental_source"]
        lyrics_settings = parameters["lyrics_settings"]
        custom_lyrics = parameters["custom_lyrics"]
        style_references = parameters["style_references"]
        reference_usage_mode = parameters["reference_usage_mode"]
        pitch = parameters["pitch"]
        tempo = parameters["tempo"]
        styles_themes = parameters["styles_themes"]
        styles_themes_to_avoid = parameters["styles_themes_to_avoid"]
        instruments = parameters["instruments"]
        instruments_to_avoid = parameters["instruments_to_avoid"]
        
        # Process lyrics
        lyrics = None
        if lyrics_settings == "generate_lyrics":
            jobs_db[job_id]["progress"] = 10
            jobs_db[job_id]["updated_at"] = datetime.now()
            
            lyrics_refs = []
            if style_references and style_references.get("lyrics_references"):
                lyrics_refs = style_references["lyrics_references"]
                
            lyrics = generate_lyrics(
                prompt=text_prompt,
                reference_lyrics=lyrics_refs,
                reference_mode=reference_usage_mode
            )
        elif lyrics_settings == "custom_lyrics" and custom_lyrics:
            lyrics = custom_lyrics
            
        jobs_db[job_id]["progress"] = 30
        jobs_db[job_id]["updated_at"] = datetime.now()
        
        # Process vocals
        vocal_track_path = None
        if vocal_settings in ["with_vocals", "only_vocals"] and generation_type in ["instrumental_vocal", "vocal_only"]:
            if vocals_source == "generate_vocals" and lyrics:
                vocal_refs = []
                if style_references and style_references.get("vocals_references"):
                    vocal_refs = style_references["vocals_references"]
                    
                vocal_track_path = generate_vocals(
                    lyrics=lyrics,
                    pitch=pitch,
                    reference_tracks=vocal_refs,
                    reference_mode=reference_usage_mode,
                    user_id=user_id
                )
            # For custom vocals input, we would retrieve from storage
            
        jobs_db[job_id]["progress"] = 60
        jobs_db[job_id]["updated_at"] = datetime.now()
        
        # Process instrumental
        instrumental_track_path = None
        if vocal_settings in ["with_vocals", "no_vocals"] and generation_type in ["instrumental", "instrumental_vocal"]:
            if instrumental_source == "generate_track":
                music_refs = []
                if style_references and style_references.get("music_references"):
                    music_refs = style_references["music_references"]
                    
                instrumental_track_path = InstrumentalGenerator.create_midi(
                    prompt=text_prompt,
                    tempo=tempo,
                    desired_instruments=instruments,
                    avoided_instruments=instruments_to_avoid,
                    styles=styles_themes,
                    avoided_styles=styles_themes_to_avoid,
                    reference_tracks=music_refs,
                    reference_mode=reference_usage_mode,
                    user_id=user_id
                )
            # For custom instrumental input, we would retrieve from storage
            
        jobs_db[job_id]["progress"] = 80
        jobs_db[job_id]["updated_at"] = datetime.now()
        
        # Combine tracks if needed
        final_track_path = None
        if vocal_track_path and instrumental_track_path:
            final_track_path = MusicGenerator.combine_tracks(
                vocals_path=vocal_track_path,
                instrumental_path=instrumental_track_path,
                user_id=user_id,
                track_id=job_id
            )
        elif vocal_track_path:
            final_track_path = vocal_track_path
        elif instrumental_track_path:
            final_track_path = instrumental_track_path
            
        jobs_db[job_id]["progress"] = 90
        jobs_db[job_id]["updated_at"] = datetime.now()
        
        # If no track was generated, raise error
        if not final_track_path:
            raise Exception("No audio could be generated with the provided settings")
        
        # Upload the file to storage
        storage_path = f"users/{user_id}/tracks/{job_id}/final.mp3"
        audio_url = storage_config.upload_file(
            local_path=final_track_path,
            storage_path=storage_path
        )
        
        # Generate waveform data
        waveform_data = MusicGenerator.generate_waveform_data(final_track_path)
        
        # Build track metadata
        track_metadata = {
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,
            "track_id": job_id,
            "prompt": text_prompt,
            "generation_type": generation_type,
            "lyrics_mode": lyrics_settings,
            "vocal_mode": vocal_settings,
            "has_lyrics": lyrics_settings != "no_lyrics",
            "has_vocals": vocal_settings != "no_vocals",
            "has_instrumental": vocal_settings != "only_vocals",
            "tempo": tempo,
            "pitch": pitch,
            "styles_themes": styles_themes,
            "instruments": instruments
        }
        
        # Store result
        result = {
            "track_url": audio_url,
            "track_metadata": track_metadata,
            "waveform_data": waveform_data,
            "lyrics": lyrics
        }
        
        # Update job with completed status and results
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["updated_at"] = datetime.now()
        jobs_db[job_id]["result"] = result
        
    except Exception as e:
        # Update job with failed status and error message
        print(f"Error processing music generation job: {str(e)}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["updated_at"] = datetime.now()
        jobs_db[job_id]["error"] = str(e)
        
class JobStatusResponse(BaseModel):
    status: str
    progress: float
    estimated_time: Optional[int] = None  # in seconds
    message: Optional[str] = None

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Check the status of a music generation job.
    
    Returns the current status, progress percentage, and estimated time remaining.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if job exists
        if job_id not in jobs_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID {job_id} not found"
            )
        
        # Check if job belongs to user
        job_data = jobs_db[job_id]
        if job_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this job"
            )
        
        # Get status information
        job_status = job_data["status"]
        progress = job_data["progress"]
        
        # Calculate estimated time remaining
        estimated_time = None
        message = None
        
        if job_status == "queued":
            message = "Your music generation job is in the queue and will be processed soon"
            if "estimated_completion_time" in job_data:
                time_delta = (job_data["estimated_completion_time"] - datetime.now()).total_seconds()
                estimated_time = max(0, int(time_delta))
        
        elif job_status == "processing":
            if progress > 0:
                # Estimate time based on progress
                elapsed_time = (datetime.now() - job_data["created_at"]).total_seconds()
                if progress < 100:
                    estimated_time = int((elapsed_time / progress) * (100 - progress))
                    message = f"Processing your music: currently at {progress}% completion"
                else:
                    estimated_time = 0
                    message = "Processing almost complete"
            else:
                message = "Processing has started but progress cannot be estimated yet"
                estimated_time = None
        
        elif job_status == "completed":
            message = "Your music has been successfully generated"
            estimated_time = 0
        
        elif job_status == "failed":
            message = f"Music generation failed: {job_data.get('error', 'Unknown error')}"
            estimated_time = 0
        
        return JobStatusResponse(
            status=job_status,
            progress=progress,
            estimated_time=estimated_time,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )

class GenerationResultResponse(BaseModel):
    track_url: str
    track_metadata: Dict[str, Any]
    waveform_data: List[float]
    lyrics: Optional[str] = None

@router.get("/jobs/{job_id}/result", response_model=GenerationResultResponse)
async def get_job_result(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve the completed music generation results.
    
    Returns the generated audio URL, track metadata, waveform data, and lyrics (if applicable).
    """
    try:
        user_id = current_user["uid"]
        
        # Check if job exists
        if job_id not in jobs_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID {job_id} not found"
            )
        
        # Check if job belongs to user
        job_data = jobs_db[job_id]
        if job_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this job"
            )
        
        # Check if job is completed
        if job_data["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is not completed yet. Current status: {job_data['status']}"
            )
        
        # Check if result exists
        if not job_data.get("result"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Result data is missing for this job"
            )
        
        # Return the result
        return GenerationResultResponse(**job_data["result"])
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving job result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job result: {str(e)}"
        )

@staticmethod
def generate_waveform_data(audio_path, num_points=100):
    """
    Generate waveform data for visualization from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        num_points (int): Number of data points to generate
        
    Returns:
        List[float]: Array of amplitude values for visualization
    """
    try:
        import numpy as np
        from pydub import AudioSegment
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to numpy array (mono)
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Normalize
        samples = samples / np.max(np.abs(samples))
        
        # Resample to desired number of points
        samples_length = len(samples)
        points_per_sample = samples_length // num_points
        
        waveform_data = []
        for i in range(num_points):
            start = i * points_per_sample
            end = min(start + points_per_sample, samples_length)
            if start < samples_length:
                # Use absolute max value in this segment
                waveform_data.append(float(np.max(np.abs(samples[start:end]))))
        
        return waveform_data
    except Exception as e:
        print(f"Error generating waveform data: {str(e)}")
        # Return a flat waveform if there's an error
        return [0.5] * num_points
    
    # Session storage (replace with database in production)
generation_sessions = {}

# Request/Response models for Step 1
class Step1Request(BaseModel):
    text_prompt: str = Field(..., description="Textual description of desired music")
    vocal_settings: VocalSettingsEnum = Field(..., description="Vocal preferences")
    lyrics_settings: LyricsSettingsEnum = Field(..., description="Lyrics preferences")
    custom_lyrics: Optional[str] = Field(None, description="Text content when custom lyrics are selected")

class Step1Response(BaseModel):
    session_id: str
    next_step: Dict[str, Any]

# Request/Response models for Step 2
class StyleReferences(BaseModel):
    lyrics_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for lyrics references")
    vocals_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for vocal references") 
    music_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for music references")

class Step2Request(BaseModel):
    style_references: StyleReferences
    reference_usage_mode: ReferenceUsageModeEnum = Field("guidance_only", description="How references should be used")

class StepResponse(BaseModel):
    session_id: str
    session_data: Dict[str, Any]
    next_step: Dict[str, Any]

# Request/Response models for Step 3
class Step3Request(BaseModel):
    instruments: List[str] = Field(default=[], description="Desired instruments")
    instruments_to_avoid: List[str] = Field(default=[], description="Instruments to exclude")
    styles_themes: List[str] = Field(default=[], description="Desired styles/themes")
    styles_themes_to_avoid: List[str] = Field(default=[], description="Styles/themes to avoid")
    pitch: Optional[str] = Field(None, description="Desired pitch (e.g., 'C#')")
    tempo: Optional[int] = Field(None, description="Desired tempo in BPM (e.g., 120)")

# Request/Response models for Step 4
class Step4Request(BaseModel):
    lyrics: Optional[str] = Field(None, description="Modified lyrics")
    regenerate_lyrics: Optional[bool] = Field(False, description="Request new lyrics generation")

class Step4Response(BaseModel):
    session_id: str
    lyrics: str
    session_data: Dict[str, Any]
    next_step: Dict[str, Any]

# Response model for Step 5
class Step5Response(BaseModel):
    job_id: str
    status: str

@router.post("/step1", response_model=Step1Response)
async def step1_basic_info(
    request: Step1Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Step 1: Submit initial generation parameters.
    
    This endpoint collects basic information about the desired music generation,
    including text prompt, vocal settings, and lyrics preferences.
    """
    try:
        user_id = current_user["uid"]
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Initialize session data
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "step": 1,
            "text_prompt": request.text_prompt,
            "vocal_settings": request.vocal_settings,
            "lyrics_settings": request.lyrics_settings,
            "custom_lyrics": request.custom_lyrics,
        }
        
        # Generate lyrics if requested
        if request.lyrics_settings == "generate_lyrics":
            try:
                lyrics = generate_lyrics(prompt=request.text_prompt)
                session_data["generated_lyrics"] = lyrics
            except Exception as e:
                print(f"Error generating lyrics: {str(e)}")
                session_data["lyrics_generation_error"] = str(e)
        elif request.lyrics_settings == "custom_lyrics":
            session_data["generated_lyrics"] = request.custom_lyrics
        
        # Store session data
        generation_sessions[session_id] = session_data
        
        # Define next step information
        next_step = {
            "step": 2,
            "endpoint": f"/api/music/step2/{session_id}",
            "description": "Submit style references for the generation job",
            "required_fields": ["style_references", "reference_usage_mode"]
        }
        
        return Step1Response(
            session_id=session_id,
            next_step=next_step
        )
        
    except Exception as e:
        print(f"Error in step 1: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step 1: {str(e)}"
        )

@router.post("/step2/{session_id}", response_model=StepResponse)
async def step2_style_references(
    session_id: str,
    request: Step2Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Step 2: Submit style references for the generation job.
    
    This endpoint collects reference files that will influence the style of the generated music.
    """
    try:
        user_id = current_user["uid"]
        
        # Verify session exists
        if session_id not in generation_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found. Please start from step 1."
            )
        
        # Verify session belongs to user
        session_data = generation_sessions[session_id]
        if session_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. This session belongs to another user."
            )
        
        # Verify correct step
        if session_data["step"] != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid step sequence. Current step is {session_data['step']}."
            )
        
        # Update session data
        session_data["step"] = 2
        session_data["updated_at"] = datetime.now().isoformat()
        session_data["style_references"] = request.style_references.dict()
        session_data["reference_usage_mode"] = request.reference_usage_mode
        
        # Validate file references exist
        # In a real implementation, you would verify the file IDs here
        
        # Define next step information
        next_step = {
            "step": 3,
            "endpoint": f"/api/music/step3/{session_id}",
            "description": "Submit musical attributes",
            "required_fields": ["instruments", "styles_themes", "pitch", "tempo"]
        }
        
        return StepResponse(
            session_id=session_id,
            session_data=session_data,
            next_step=next_step
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in step 2: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step 2: {str(e)}"
        )

@router.post("/step3/{session_id}", response_model=StepResponse)
async def step3_musical_attributes(
    session_id: str,
    request: Step3Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Step 3: Submit musical attributes.
    
    This endpoint collects specific musical attributes like instruments, styles, tempo, and pitch.
    """
    try:
        user_id = current_user["uid"]
        
        # Verify session exists
        if session_id not in generation_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found. Please start from step 1."
            )
        
        # Verify session belongs to user
        session_data = generation_sessions[session_id]
        if session_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. This session belongs to another user."
            )
        
        # Verify correct step
        if session_data["step"] != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid step sequence. Current step is {session_data['step']}."
            )
        
        # Update session data
        session_data["step"] = 3
        session_data["updated_at"] = datetime.now().isoformat()
        session_data["instruments"] = request.instruments
        session_data["instruments_to_avoid"] = request.instruments_to_avoid
        session_data["styles_themes"] = request.styles_themes
        session_data["styles_themes_to_avoid"] = request.styles_themes_to_avoid
        session_data["pitch"] = request.pitch
        session_data["tempo"] = request.tempo
        
        # Define next step
        next_step = {
            "step": 4,
            "endpoint": f"/api/music/step4/{session_id}",
            "description": "Review and potentially modify lyrics",
            "required_fields": []
        }
        
        # Skip lyrics step if no lyrics are requested
        if session_data["lyrics_settings"] == "no_lyrics":
            next_step = {
                "step": 5,
                "endpoint": f"/api/music/step5/{session_id}/finalize",
                "description": "Finalize all parameters and start generation",
                "required_fields": []
            }
        
        return StepResponse(
            session_id=session_id,
            session_data=session_data,
            next_step=next_step
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in step 3: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step 3: {str(e)}"
        )

@router.post("/step4/{session_id}", response_model=Step4Response)
async def step4_lyrics_review(
    session_id: str,
    request: Step4Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Step 4: Review and potentially modify generated lyrics.
    
    This endpoint allows users to review, modify, or regenerate lyrics.
    """
    try:
        user_id = current_user["uid"]
        
        # Verify session exists
        if session_id not in generation_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found. Please start from step 1."
            )
        
        # Verify session belongs to user
        session_data = generation_sessions[session_id]
        if session_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. This session belongs to another user."
            )
        
        # Verify correct step
        if session_data["step"] != 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid step sequence. Current step is {session_data['step']}."
            )
        
        # Check if lyrics settings allows for lyrics
        if session_data["lyrics_settings"] == "no_lyrics":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lyrics review not applicable - lyrics setting is 'no_lyrics'."
            )
        
        # Update session data
        session_data["step"] = 4
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Handle lyrics modification/regeneration
        lyrics = session_data.get("generated_lyrics", "")
        
        if request.regenerate_lyrics:
            # Regenerate lyrics
            try:
                lyrics = generate_lyrics(prompt=session_data["text_prompt"])
                session_data["generated_lyrics"] = lyrics
                session_data["lyrics_regenerated"] = True
            except Exception as e:
                print(f"Error regenerating lyrics: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to regenerate lyrics: {str(e)}"
                )
        elif request.lyrics is not None:
            # Use modified lyrics
            lyrics = request.lyrics
            session_data["generated_lyrics"] = lyrics
            session_data["lyrics_modified"] = True
        
        # Define next step
        next_step = {
            "step": 5,
            "endpoint": f"/api/music/step5/{session_id}/finalize",
            "description": "Finalize all parameters and start generation",
            "required_fields": []
        }
        
        return Step4Response(
            session_id=session_id,
            lyrics=lyrics,
            session_data=session_data,
            next_step=next_step
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in step 4: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step 4: {str(e)}"
        )

@router.post("/step5/{session_id}/finalize", response_model=Step5Response)
async def step5_finalize(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Step 5: Finalize all parameters and start the generation job.
    
    This endpoint finalizes the music generation request and initiates the
    asynchronous generation process.
    """
    try:
        user_id = current_user["uid"]
        
        # Verify session exists
        if session_id not in generation_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found. Please start from step 1."
            )
        
        # Verify session belongs to user
        session_data = generation_sessions[session_id]
        if session_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. This session belongs to another user."
            )
        
        # Verify correct step (either 3 if skipped lyrics, or 4)
        if session_data["step"] not in [3, 4]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid step sequence. Current step is {session_data['step']}."
            )
        
        # Generate a job ID (use session ID for traceability)
        job_id = f"job-{session_id}"
        
        # Convert session data to job parameters
        job_params = {
            "job_id": job_id,
            "user_id": user_id,
            "text_prompt": session_data["text_prompt"],
            "vocal_settings": session_data["vocal_settings"],
            "lyrics_settings": session_data["lyrics_settings"],
            "custom_lyrics": session_data.get("generated_lyrics"),
            "style_references": session_data.get("style_references", {}),
            "reference_usage_mode": session_data.get("reference_usage_mode", "guidance_only"),
            "instruments": session_data.get("instruments", []),
            "instruments_to_avoid": session_data.get("instruments_to_avoid", []),
            "styles_themes": session_data.get("styles_themes", []),
            "styles_themes_to_avoid": session_data.get("styles_themes_to_avoid", []),
            "pitch": session_data.get("pitch"),
            "tempo": session_data.get("tempo"),
            "created_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        # In a real implementation, you'd store this in a proper job queue or database
        # For demonstration, we'll use a background task
        background_tasks.add_task(
            start_generation_job,
            job_params=job_params
        )
        
        # Clean up session (optional, can also keep for reference)
        # del generation_sessions[session_id]
        
        return Step5Response(
            job_id=job_id,
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in step 5: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step 5: {str(e)}"
        )

async def start_generation_job(job_params: dict):
    """
    Background task to start the generation job.
    
    This would typically interact with your existing music generation system.
    """
    # This is where you would call your music generation pipeline
    # You can use the same process as in your /api/music/generate/init endpoint
    
    # For demonstration purposes, here's a placeholder
    try:
        # Simulate processing time
        import asyncio
        await asyncio.sleep(2)
        
        # Update job status (in a real implementation, this would update a database)
        job_params["status"] = "processing"
        
        # Additional processing would happen here...
        
        print(f"Started generation job: {job_params['job_id']}")
        
    except Exception as e:
        print(f"Error starting generation job: {str(e)}")
        # Update job status to failed
        job_params["status"] = "failed"
        job_params["error"] = str(e)

# In-memory lyrics storage (replace with database in production)
lyrics_db = {}

# Request and Response Models for Lyrics Generation
class LyricsGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description for lyrics theme")
    style: str = Field(..., description="Lyrical style (e.g., 'poetic', 'rap')")
    reference_lyrics: Optional[List[str]] = Field(default=None, description="Optional reference lyrics")

class LyricsGenerationResponse(BaseModel):
    lyrics_id: str
    lyrics_text: str
    status: str
    created_at: datetime
    metadata: Dict[str, Any]

# Request and Response Models for Lyrics Modification
class LyricsModificationRequest(BaseModel):
    modified_text: str = Field(..., description="Updated lyrics text")

class LyricsModificationResponse(BaseModel):
    lyrics_id: str
    lyrics_text: str
    status: str
    updated_at: datetime
    version: int
    metadata: Dict[str, Any]

@router.post("/generate", response_model=LyricsGenerationResponse)
async def generate_lyrics_endpoint(
    request: LyricsGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate lyrics based on prompt and style.
    
    This endpoint creates new lyrics based on the given theme and style preferences.
    """
    try:
        user_id = current_user["uid"]
        
        # Generate a unique lyrics ID
        lyrics_id = str(uuid.uuid4())
        
        # Generate lyrics based on the prompt and style
        try:
            # Call the lyrics generation function
            lyrics_text = generate_lyrics(
                prompt=request.prompt,
                style=request.style,
                reference_lyrics=request.reference_lyrics
            )
        except Exception as e:
            print(f"Error generating lyrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate lyrics: {str(e)}"
            )
        
        # Create metadata
        metadata = {
            "user_id": user_id,
            "prompt": request.prompt,
            "style": request.style,
            "has_references": request.reference_lyrics is not None and len(request.reference_lyrics) > 0,
            "reference_count": len(request.reference_lyrics) if request.reference_lyrics else 0,
        }
        
        # Create timestamp
        created_at = datetime.now()
        
        # Store the lyrics data (in a real app, this would go to a database)
        lyrics_data = {
            "lyrics_id": lyrics_id,
            "lyrics_text": lyrics_text,
            "user_id": user_id,
            "created_at": created_at,
            "updated_at": created_at,
            "version": 1,
            "metadata": metadata,
            "history": [
                {
                    "version": 1,
                    "text": lyrics_text,
                    "timestamp": created_at
                }
            ]
        }
        
        lyrics_db[lyrics_id] = lyrics_data
        
        return LyricsGenerationResponse(
            lyrics_id=lyrics_id,
            lyrics_text=lyrics_text,
            status="success",
            created_at=created_at,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in lyrics generation endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process lyrics generation: {str(e)}"
        )

@router.put("/{lyrics_id}/modify", response_model=LyricsModificationResponse)
async def modify_lyrics(
    lyrics_id: str,
    request: LyricsModificationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Modify existing lyrics.
    
    This endpoint updates previously generated lyrics with new text.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if lyrics exist
        if lyrics_id not in lyrics_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lyrics with ID {lyrics_id} not found"
            )
        
        # Check if lyrics belong to user
        lyrics_data = lyrics_db[lyrics_id]
        if lyrics_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to modify these lyrics"
            )
        
        # Update lyrics
        updated_at = datetime.now()
        new_version = lyrics_data["version"] + 1
        
        # Add to history
        lyrics_data["history"].append({
            "version": new_version,
            "text": request.modified_text,
            "timestamp": updated_at
        })
        
        # Update main data
        lyrics_data["lyrics_text"] = request.modified_text
        lyrics_data["updated_at"] = updated_at
        lyrics_data["version"] = new_version
        lyrics_data["metadata"]["modified"] = True
        
        return LyricsModificationResponse(
            lyrics_id=lyrics_id,
            lyrics_text=request.modified_text,
            status="success",
            updated_at=updated_at,
            version=new_version,
            metadata=lyrics_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error modifying lyrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to modify lyrics: {str(e)}"
        )

# Optional: Add a GET endpoint to retrieve lyrics by ID
@router.get("/{lyrics_id}", response_model=LyricsGenerationResponse)
async def get_lyrics(
    lyrics_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve previously generated lyrics.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if lyrics exist
        if lyrics_id not in lyrics_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lyrics with ID {lyrics_id} not found"
            )
        
        # Check if lyrics belong to user
        lyrics_data = lyrics_db[lyrics_id]
        if lyrics_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access these lyrics"
            )
        
        return LyricsGenerationResponse(
            lyrics_id=lyrics_data["lyrics_id"],
            lyrics_text=lyrics_data["lyrics_text"],
            status="success",
            created_at=lyrics_data["created_at"],
            metadata=lyrics_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving lyrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve lyrics: {str(e)}"
        )
        # In-memory vocals storage (replace with database in production)
        
vocals_db = {}

# Request and Response Models for Vocals Generation
class VocalStylePreferences(BaseModel):
    voice_type: Optional[str] = Field(None, description="Type of voice (e.g., 'male', 'female', 'child')")
    emotion: Optional[str] = Field(None, description="Emotional tone (e.g., 'cheerful', 'sad', 'angry')")
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Intensity of the emotional tone (0.0-1.0)")
    clarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Voice clarity parameter (0.0-1.0)")
    stability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Voice stability parameter (0.0-1.0)")
    accent: Optional[str] = Field(None, description="Accent preference (e.g., 'american', 'british', 'australian')")

class PitchAdjustments(BaseModel):
    base_pitch: Optional[str] = Field(None, description="Base pitch setting (e.g., 'C4', 'A3')")
    pitch_range: Optional[float] = Field(None, ge=0.1, le=2.0, description="Range of pitch variation (0.1-2.0)")
    contour: Optional[List[str]] = Field(None, description="Pitch contour points (e.g., '(0%,+20Hz) (50%,-10Hz)')")

class VocalsGenerationRequest(BaseModel):
    lyrics_id: str = Field(..., description="ID of lyrics to vocalize")
    style: Optional[VocalStylePreferences] = Field(None, description="Vocal style preferences")
    pitch: Optional[PitchAdjustments] = Field(None, description="Vocal pitch adjustments")

class VocalsGenerationResponse(BaseModel):
    vocals_id: str
    vocals_url: Optional[str]
    status: str
    estimated_completion_time: Optional[int] = None

# Request and Response Models for Vocals Customization
class VocalsCustomizationRequest(BaseModel):
    pitch_adjustment: Optional[PitchAdjustments] = Field(None, description="Changes to vocal pitch")
    style_adjustment: Optional[VocalStylePreferences] = Field(None, description="Changes to vocal style")

class VocalsCustomizationResponse(BaseModel):
    vocals_id: str
    vocals_url: Optional[str]
    status: str
    metadata: Dict[str, Any]

@router.post("/generate", response_model=VocalsGenerationResponse)
async def generate_vocals_endpoint(
    request: VocalsGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate vocals from lyrics.
    
    This endpoint creates vocals based on lyrics and style preferences.
    """
    try:
        user_id = current_user["uid"]
        
        # Validate lyrics existence (in a real implementation, this would check your database)
        lyrics_text = get_lyrics_by_id(request.lyrics_id, user_id)
        
        if not lyrics_text:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lyrics with ID {request.lyrics_id} not found or does not belong to you"
            )
        
        # Generate a unique vocals ID
        vocals_id = str(uuid.uuid4())
        
        # Create initial vocals record
        vocals_data = {
            "vocals_id": vocals_id,
            "user_id": user_id,
            "lyrics_id": request.lyrics_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status": "processing",
            "vocals_url": None,
            "style": request.style.dict() if request.style else {},
            "pitch": request.pitch.dict() if request.pitch else {},
            "metadata": {
                "lyrics_text": lyrics_text[:100] + "..." if len(lyrics_text) > 100 else lyrics_text,
                "version": 1,
                "processing_history": []
            }
        }
        
        # Store initial vocals data
        vocals_db[vocals_id] = vocals_data
        
        # Start vocals generation in background
        background_tasks.add_task(
            process_vocals_generation,
            vocals_id=vocals_id,
            user_id=user_id,
            lyrics_text=lyrics_text,
            style=request.style.dict() if request.style else {},
            pitch=request.pitch.dict() if request.pitch else {}
        )
        
        return VocalsGenerationResponse(
            vocals_id=vocals_id,
            vocals_url=None,
            status="processing",
            estimated_completion_time=calculate_estimated_time(lyrics_text, request.style, request.pitch)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in vocals generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process vocals generation: {str(e)}"
        )

@router.put("/{vocals_id}/customize", response_model=VocalsCustomizationResponse)
async def customize_vocals_endpoint(
    vocals_id: str,
    request: VocalsCustomizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Adjust parameters of generated vocals.
    
    This endpoint allows customizing pitch and style of already generated vocals.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if vocals exist
        if vocals_id not in vocals_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vocals with ID {vocals_id} not found"
            )
        
        # Check if vocals belong to user
        vocals_data = vocals_db[vocals_id]
        if vocals_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to customize these vocals"
            )
        
        # Check if vocals are in a customizable state
        if vocals_data["status"] == "processing":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vocals are still processing and cannot be customized yet"
            )
        
        if vocals_data["status"] == "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed vocals cannot be customized. Please generate new vocals."
            )
        
        # Update vocals with new parameters
        new_version = vocals_data["metadata"]["version"] + 1
        
        # Store original values before updating
        previous_style = vocals_data["style"]
        previous_pitch = vocals_data["pitch"]
        
        # Update style if provided
        if request.style_adjustment:
            for key, value in request.style_adjustment.dict(exclude_unset=True).items():
                if value is not None:
                    vocals_data["style"][key] = value
        
        # Update pitch if provided
        if request.pitch_adjustment:
            for key, value in request.pitch_adjustment.dict(exclude_unset=True).items():
                if value is not None:
                    vocals_data["pitch"][key] = value
        
        # Record the customization in processing history
        vocals_data["metadata"]["processing_history"].append({
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "type": "customization",
            "previous_style": previous_style,
            "previous_pitch": previous_pitch,
            "new_style": vocals_data["style"],
            "new_pitch": vocals_data["pitch"]
        })
        
        # Update metadata version
        vocals_data["metadata"]["version"] = new_version
        
        # Mark as processing
        vocals_data["status"] = "processing"
        vocals_data["updated_at"] = datetime.now()
        
        # Start vocals customization in background
        background_tasks.add_task(
            process_vocals_customization,
            vocals_id=vocals_id,
            user_id=user_id,
            original_vocals_url=vocals_data["vocals_url"],
            style=vocals_data["style"],
            pitch=vocals_data["pitch"]
        )
        
        return VocalsCustomizationResponse(
            vocals_id=vocals_id,
            vocals_url=None,  # Will be updated when processing completes
            status="processing",
            metadata=vocals_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in vocals customization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process vocals customization: {str(e)}"
        )

# Helper function to get lyrics by ID (replace with database implementation)
def get_lyrics_by_id(lyrics_id: str, user_id: str) -> Optional[str]:
    """
    Retrieve lyrics text by ID.
    In a real implementation, this would query your database.
    """
    # This is a placeholder implementation
    if lyrics_id == "sample-lyrics-id":
        return "These are sample lyrics for demonstration purposes. In a real implementation, these would be retrieved from your database."
    # In a real implementation, query your database:
    # lyrics = db.lyrics.find_one({"lyrics_id": lyrics_id, "user_id": user_id})
    # return lyrics["lyrics_text"] if lyrics else None
    return None

# Background task to process vocals generation
async def process_vocals_generation(vocals_id: str, user_id: str, lyrics_text: str, style: dict, pitch: dict):
    """
    Background task to generate vocals.
    This runs asynchronously after the API response is sent.
    """
    try:
        # Update status to processing
        vocals_db[vocals_id]["status"] = "processing"
        vocals_db[vocals_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "processing_started"
        })
        
        # Call your vocals generation function
        vocal_track_path = generate_vocals(
            lyrics=lyrics_text,
            voice_type=style.get("voice_type"),
            emotion=style.get("emotion"),
            intensity=style.get("intensity"),
            clarity=style.get("clarity"),
            stability=style.get("stability"),
            accent=style.get("accent"),
            base_pitch=pitch.get("base_pitch"),
            pitch_range=pitch.get("pitch_range"),
            contour=pitch.get("contour"),
            user_id=user_id
        )
        
        # Upload the file to storage
        storage_path = f"users/{user_id}/vocals/{vocals_id}.mp3"
        vocals_url = storage_config.upload_file(
            local_path=vocal_track_path,
            storage_path=storage_path
        )
        
        # Update the vocals record with completed status and URL
        vocals_db[vocals_id]["status"] = "completed"
        vocals_db[vocals_id]["vocals_url"] = vocals_url
        vocals_db[vocals_id]["updated_at"] = datetime.now()
        vocals_db[vocals_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        # Update with failed status
        vocals_db[vocals_id]["status"] = "failed"
        vocals_db[vocals_id]["updated_at"] = datetime.now()
        vocals_db[vocals_id]["metadata"]["error"] = str(e)
        vocals_db[vocals_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        })
        print(f"Error processing vocals generation: {str(e)}")

# Background task to process vocals customization
async def process_vocals_customization(vocals_id: str, user_id: str, original_vocals_url: str, style: dict, pitch: dict):
    """
    Background task to customize vocals.
    This runs asynchronously after the API response is sent.
    """
    try:
        # Update status to processing
        vocals_db[vocals_id]["status"] = "processing"
        vocals_db[vocals_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "customization_started"
        })
        
        # Call your vocals customization function
        vocal_track_path = customize_vocals(
            original_vocals_url=original_vocals_url,
            voice_type=style.get("voice_type"),
            emotion=style.get("emotion"),
            intensity=style.get("intensity"),
            clarity=style.get("clarity"),
            stability=style.get("stability"),
            accent=style.get("accent"),
            base_pitch=pitch.get("base_pitch"),
            pitch_range=pitch.get("pitch_range"),
            contour=pitch.get("contour"),
            user_id=user_id
        )
        
        # Upload the file to storage
        storage_path = f"users/{user_id}/vocals/{vocals_id}_v{vocals_db[vocals_id]['metadata']['version']}.mp3"
        vocals_url = storage_config.upload_file(
            local_path=vocal_track_path,
            storage_path=storage_path
        )
        
        # Update the vocals record with completed status and URL
        vocals_db[vocals_id]["status"] = "completed"
        vocals_db[vocals_id]["vocals_url"] = vocals_url
        vocals_db[vocals_id]["updated_at"] = datetime.now()
        vocals_db[vocals_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "customization_completed"
        })
        
    except Exception as e:
        # Update with failed status
        vocals_db[vocals_id]["status"] = "failed"
        vocals_db[vocals_id]["updated_at"] = datetime.now()
        vocals_db[vocals_id]["metadata"]["error"] = str(e)
        vocals_db[vocals_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "customization_failed",
            "error": str(e)
        })
        print(f"Error processing vocals customization: {str(e)}")

# Helper function to calculate estimated processing time
def calculate_estimated_time(lyrics_text: str, style: Optional[VocalStylePreferences], pitch: Optional[PitchAdjustments]) -> int:
    """
    Calculate estimated time for vocals generation in seconds.
    """
    # Basic estimate based on lyrics length
    lyrics_length = len(lyrics_text)
    base_time = max(30, lyrics_length // 10)  # Minimum 30 seconds, plus 1 second per 10 characters
    
    # Adjust based on style complexity
    style_multiplier = 1.0
    if style:
        if style.emotion and style.emotion not in ["neutral", "normal"]:
            style_multiplier *= 1.2  # Emotional styles take longer
        if style.intensity and style.intensity > 0.7:
            style_multiplier *= 1.1  # Higher intensity takes longer
    
    # Adjust based on pitch complexity
    pitch_multiplier = 1.0
    if pitch:
        if pitch.contour:
            pitch_multiplier *= 1.3  # Complex pitch contours take longer
        if pitch.pitch_range and pitch.pitch_range > 1.5:
            pitch_multiplier *= 1.2  # Wider pitch ranges take longer
    
    # Calculate final estimate
    return int(base_time * style_multiplier * pitch_multiplier)

# Additional endpoint to get vocals status (optional but recommended)
@router.get("/{vocals_id}", response_model=VocalsCustomizationResponse)
async def get_vocals_status(
    vocals_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the status of a vocals generation or customization process.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if vocals exist
        if vocals_id not in vocals_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vocals with ID {vocals_id} not found"
            )
        
        # Check if vocals belong to user
        vocals_data = vocals_db[vocals_id]
        if vocals_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access these vocals"
            )
        
        return VocalsCustomizationResponse(
            vocals_id=vocals_id,
            vocals_url=vocals_data.get("vocals_url"),
            status=vocals_data["status"],
            metadata=vocals_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving vocals status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve vocals status: {str(e)}")
        
        
# In-memory storage for instrumental tracks (replace with database in production)
instrumentals_db = {}

# Request and Response Models for Instrumental Generation
class StyleReference(BaseModel):
    file_id: str
    weight: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="How strongly this reference influences the result (0.0-2.0)")

class InstrumentalGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of desired instrumental")
    style_references: Optional[List[StyleReference]] = Field(default=[], description="Reference tracks to influence style")
    instruments: Optional[List[str]] = Field(default=[], description="Desired instruments to include")
    tempo: Optional[int] = Field(None, ge=40, le=240, description="Desired tempo in BPM (40-240)")
    pitch: Optional[str] = Field(None, description="Desired key/pitch (e.g., 'C Major', 'F# Minor')")

class InstrumentalGenerationResponse(BaseModel):
    instrumental_id: str
    instrumental_url: Optional[str] = None
    status: str
    estimated_completion_time: Optional[int] = None
    metadata: Dict[str, Any]

# Request and Response Models for Instrumental Customization
class InstrumentAdjustment(BaseModel):
    instrument: str
    volume: Optional[float] = Field(None, ge=0.0, le=2.0, description="Volume adjustment (0.0-2.0)")
    pan: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Pan adjustment (-1.0 left to 1.0 right)")
    effects: Optional[Dict[str, float]] = Field(None, description="Effects adjustments (e.g., reverb, delay)")

class InstrumentalCustomizationRequest(BaseModel):
    instrument_adjustments: Optional[List[InstrumentAdjustment]] = Field(default=[], description="Changes to instrument mix")
    tempo_adjustment: Optional[int] = Field(None, ge=40, le=240, description="Changes to tempo in BPM (40-240)")

class InstrumentalCustomizationResponse(BaseModel):
    instrumental_id: str
    instrumental_url: Optional[str] = None
    status: str
    metadata: Dict[str, Any]

@router.post("/generate", response_model=InstrumentalGenerationResponse)
async def generate_instrumental(
    request: InstrumentalGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate an instrumental track based on text description and parameters.
    
    This endpoint initiates the generation of an instrumental track and returns immediately.
    The actual generation process happens asynchronously in the background.
    """
    try:
        user_id = current_user["uid"]
        
        # Generate a unique ID for the instrumental
        instrumental_id = str(uuid.uuid4())
        
        # Resolve file paths for style references
        style_reference_paths = []
        if request.style_references:
            for ref in request.style_references:
                # In a real implementation, you would retrieve actual file paths from your storage
                # Here we're just simulating the process
                file_path = get_file_path_by_id(ref.file_id, user_id)
                if file_path:
                    style_reference_paths.append({
                        "path": file_path,
                        "weight": ref.weight
                    })
        
        # Create initial record
        instrumental_data = {
            "instrumental_id": instrumental_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status": "processing",
            "instrumental_url": None,
            "prompt": request.prompt,
            "instruments": request.instruments,
            "tempo": request.tempo,
            "pitch": request.pitch,
            "style_references": [ref.dict() for ref in request.style_references],
            "metadata": {
                "version": 1,
                "processing_history": [{
                    "timestamp": datetime.now().isoformat(),
                    "status": "generation_started",
                    "parameters": request.dict()
                }]
            }
        }
        
        # Store in database (using in-memory dict for now)
        instrumentals_db[instrumental_id] = instrumental_data
        
        # Calculate estimated time
        estimated_time = calculate_estimated_time(request)
        
        # Start generation in background
        background_tasks.add_task(
            process_instrumental_generation,
            instrumental_id=instrumental_id,
            user_id=user_id,
            prompt=request.prompt,
            style_reference_paths=style_reference_paths,
            instruments=request.instruments,
            tempo=request.tempo,
            pitch=request.pitch
        )
        
        return InstrumentalGenerationResponse(
            instrumental_id=instrumental_id,
            status="processing",
            estimated_completion_time=estimated_time,
            metadata=instrumental_data["metadata"]
        )
        
    except Exception as e:
        print(f"Error in instrumental generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process instrumental generation: {str(e)}"
        )

@router.put("/{instrumental_id}/customize", response_model=InstrumentalCustomizationResponse)
async def customize_instrumental(
    instrumental_id: str,
    request: InstrumentalCustomizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Customize an existing instrumental track by adjusting instruments and tempo.
    
    This endpoint allows for post-generation adjustments to instrument mix and tempo.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if instrumental exists
        if instrumental_id not in instrumentals_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrumental with ID {instrumental_id} not found"
            )
        
        # Check if instrumental belongs to user
        instrumental_data = instrumentals_db[instrumental_id]
        if instrumental_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to customize this instrumental"
            )
        
        # Check if instrumental is in a customizable state
        if instrumental_data["status"] == "processing":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Instrumental is still processing and cannot be customized yet"
            )
        
        if instrumental_data["status"] == "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed instrumental cannot be customized. Please generate a new one."
            )
        
        # Update version
        new_version = instrumental_data["metadata"]["version"] + 1
        
        # Store original values
        original_tempo = instrumental_data["tempo"]
        
        # Update tempo if provided
        if request.tempo_adjustment is not None:
            instrumental_data["tempo"] = request.tempo_adjustment
        
        # Record customization in history
        instrumental_data["metadata"]["processing_history"].append({
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "type": "customization",
            "changes": {
                "tempo": {
                    "from": original_tempo,
                    "to": instrumental_data["tempo"]
                },
                "instrument_adjustments": [adj.dict() for adj in request.instrument_adjustments]
            }
        })
        
        # Update metadata version
        instrumental_data["metadata"]["version"] = new_version
        
        # Mark as processing
        instrumental_data["status"] = "processing"
        instrumental_data["updated_at"] = datetime.now()
        
        # Start customization in background
        background_tasks.add_task(
            process_instrumental_customization,
            instrumental_id=instrumental_id,
            user_id=user_id,
            original_track_url=instrumental_data["instrumental_url"],
            instrument_adjustments=request.instrument_adjustments,
            tempo=instrumental_data["tempo"]
        )
        
        return InstrumentalCustomizationResponse(
            instrumental_id=instrumental_id,
            status="processing",
            metadata=instrumental_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in instrumental customization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process instrumental customization: {str(e)}"
        )

# Helper function to get file path by ID (replace with database implementation)
def get_file_path_by_id(file_id: str, user_id: str) -> Optional[str]:
    """
    Get the file path for a given file ID.
    In a real implementation, this would query your database.
    """
    # This is a placeholder implementation
    if file_id == "sample-file-id":
        return f"path/to/user/{user_id}/files/{file_id}.mp3"
    return None

# Background task to process instrumental generation
async def process_instrumental_generation(
    instrumental_id: str, 
    user_id: str, 
    prompt: str, 
    style_reference_paths: List[Dict[str, Any]], 
    instruments: List[str], 
    tempo: Optional[int], 
    pitch: Optional[str]
):
    """
    Background task to generate an instrumental track.
    """
    try:
        # Update status to processing
        instrumentals_db[instrumental_id]["status"] = "processing"
        instrumentals_db[instrumental_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "processing"
        })
        
        # Extract reference tracks
        reference_tracks = [ref["path"] for ref in style_reference_paths]
        reference_weights = [ref["weight"] for ref in style_reference_paths]
        
        # Call the instrumental generation function
        instrumental_track_path = InstrumentalGenerator.create_midi(
            prompt=prompt,
            tempo=tempo,
            desired_instruments=instruments,
            avoided_instruments=[],  # Could be added as a parameter if needed
            styles=[],  # Could be extracted from prompt or added as a parameter
            avoided_styles=[],  # Could be added as a parameter if needed
            reference_tracks=reference_tracks,
            reference_weights=reference_weights,
            reference_mode="guidance_only",  # Could be a parameter
            pitch=pitch,
            user_id=user_id
        )
        
        # Upload the file to storage
        storage_path = f"users/{user_id}/instrumentals/{instrumental_id}.mp3"
        instrumental_url = storage_config.upload_file(
            local_path=instrumental_track_path,
            storage_path=storage_path
        )
        
        # Update record with completed status and URL
        instrumentals_db[instrumental_id]["status"] = "completed"
        instrumentals_db[instrumental_id]["instrumental_url"] = instrumental_url
        instrumentals_db[instrumental_id]["updated_at"] = datetime.now()
        instrumentals_db[instrumental_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        # Update with failed status
        instrumentals_db[instrumental_id]["status"] = "failed"
        instrumentals_db[instrumental_id]["updated_at"] = datetime.now()
        instrumentals_db[instrumental_id]["metadata"]["error"] = str(e)
        instrumentals_db[instrumental_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        })
        print(f"Error processing instrumental generation: {str(e)}")

# Background task to process instrumental customization
async def process_instrumental_customization(
    instrumental_id: str, 
    user_id: str, 
    original_track_url: str, 
    instrument_adjustments: List[InstrumentAdjustment], 
    tempo: Optional[int]
):
    """
    Background task to customize an instrumental track.
    """
    try:
        # Update status to processing
        instrumentals_db[instrumental_id]["status"] = "processing"
        instrumentals_db[instrumental_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "customization_started"
        })
        
        # Format instrument adjustments for the generator
        formatted_adjustments = []
        for adj in instrument_adjustments:
            formatted_adjustments.append({
                "instrument": adj.instrument,
                "volume": adj.volume,
                "pan": adj.pan,
                "effects": adj.effects or {}
            })
        
        # Call the instrumental customization function
        # (Assuming a method exists in InstrumentalGenerator for this)
        customized_track_path = InstrumentalGenerator.customize_instrumental(
            original_track_url=original_track_url,
            instrument_adjustments=formatted_adjustments,
            tempo=tempo,
            user_id=user_id
        )
        
        # Upload the file to storage
        version = instrumentals_db[instrumental_id]["metadata"]["version"]
        storage_path = f"users/{user_id}/instrumentals/{instrumental_id}_v{version}.mp3"
        instrumental_url = storage_config.upload_file(
            local_path=customized_track_path,
            storage_path=storage_path
        )
        
        # Update record with completed status and URL
        instrumentals_db[instrumental_id]["status"] = "completed"
        instrumentals_db[instrumental_id]["instrumental_url"] = instrumental_url
        instrumentals_db[instrumental_id]["updated_at"] = datetime.now()
        instrumentals_db[instrumental_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "customization_completed"
        })
        
    except Exception as e:
        # Update with failed status
        instrumentals_db[instrumental_id]["status"] = "failed"
        instrumentals_db[instrumental_id]["updated_at"] = datetime.now()
        instrumentals_db[instrumental_id]["metadata"]["error"] = str(e)
        instrumentals_db[instrumental_id]["metadata"]["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": "customization_failed",
            "error": str(e)
        })
        print(f"Error processing instrumental customization: {str(e)}")

# Helper function to calculate estimated processing time
def calculate_estimated_time(request: InstrumentalGenerationRequest) -> int:
    """
    Calculate estimated time for instrumental generation in seconds.
    """
    # Base time - complex generation takes time
    base_time = 60
    
    # Adjust based on number of requested instruments
    if request.instruments:
        base_time += len(request.instruments) * 5
    
    # Adjust based on number of style references
    if request.style_references:
        base_time += len(request.style_references) * 10
    
    # More complex prompts might take longer
    prompt_length = len(request.prompt)
    if prompt_length > 100:
        base_time += 30
    
    return base_time

# Optional: Add a GET endpoint to retrieve instrumental status
@router.get("/{instrumental_id}", response_model=InstrumentalCustomizationResponse)
async def get_instrumental_status(
    instrumental_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the status of an instrumental generation or customization process.
    """
    try:
        user_id = current_user["uid"]
        
        # Check if instrumental exists
        if instrumental_id not in instrumentals_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrumental with ID {instrumental_id} not found"
            )
        
        # Check if instrumental belongs to user
        instrumental_data = instrumentals_db[instrumental_id]
        if instrumental_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this instrumental"
            )
        
        return InstrumentalCustomizationResponse(
            instrumental_id=instrumental_id,
            instrumental_url=instrumental_data.get("instrumental_url"),
            status=instrumental_data["status"],
            metadata=instrumental_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving instrumental status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve instrumental status: {str(e)}"
        )
        
# Define sort and filter options
class SortField(str, Enum):
    CREATED_AT = "created_at"
    TITLE = "title"
    DURATION = "duration"
    TYPE = "type"

class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"

class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"

# Response models
class PaginationInfo(BaseModel):
    total_items: int
    items_per_page: int
    current_page: int
    total_pages: int
    has_next: bool
    has_prev: bool

class GenerationParameters(BaseModel):
    prompt: str
    vocal_settings: str
    lyrics_settings: str
    instruments: List[str] = []
    styles_themes: List[str] = []
    tempo: Optional[int] = None
    pitch: Optional[str] = None

class TrackComponent(BaseModel):
    component_type: str  # "vocals", "instrumental", "lyrics"
    url: str
    format: str

class TrackSummary(BaseModel):
    track_id: str
    title: str
    created_at: datetime
    type: str  # "instrumental", "vocal", "full_track"
    duration: Optional[float] = None
    audio_url: str
    thumbnail_url: Optional[str] = None

class TracksListResponse(BaseModel):
    tracks: List[TrackSummary]
    pagination: PaginationInfo

class TrackDetail(BaseModel):
    track_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    type: str
    duration: Optional[float] = None
    audio_url: str
    waveform_data: List[float]
    thumbnail_url: Optional[str] = None
    generation_parameters: GenerationParameters
    components: List[TrackComponent]
    metadata: Dict[str, Any]

class DeleteResponse(BaseModel):
    status: str
    message: str

@router.get("", response_model=TracksListResponse)
async def list_user_tracks(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=50, description="Items per page"),
    sort_by: SortField = Query(SortField.CREATED_AT, description="Field to sort by"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="Sort direction"),
    filter: Optional[str] = Query(None, description="Filter criteria (format: field:operator:value,...)"),
    current_user: dict = Depends(get_current_user)
):
    """
    List all tracks generated by the user with pagination, sorting, and filtering options.
    
    The filter parameter accepts a comma-separated list of conditions in the format:
    field:operator:value
    
    Examples:
    - type:eq:instrumental (tracks of type 'instrumental')
    - created_at:gt:2025-01-01 (tracks created after Jan 1, 2025)
    - title:contains:summer (tracks with 'summer' in the title)
    """
    try:
        user_id = current_user["uid"]
        
        # Process filter conditions
        filter_conditions = []
        if filter:
            for condition in filter.split(","):
                parts = condition.split(":")
                if len(parts) == 3:
                    field, operator, value = parts
                    filter_conditions.append({
                        "field": field,
                        "operator": operator,
                        "value": value
                    })
        
        # In a real implementation, you would query your database with these parameters
        # Here, we'll simulate the response
        
        # Get total count (would come from database in real implementation)
        total_items = 25  # Simulated count
        
        # Calculate pagination info
        total_pages = (total_items + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1
        
        # Create pagination info
        pagination = PaginationInfo(
            total_items=total_items,
            items_per_page=limit,
            current_page=page,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
        # Simulate tracks (in real implementation, these would come from the database)
        tracks = [
            TrackSummary(
                track_id=f"track-{i}",
                title=f"Summer Beat {i}",
                created_at=datetime.now(),
                type="full_track",
                duration=180.5,
                audio_url=f"https://storage.example.com/users/{user_id}/tracks/track-{i}.mp3",
                thumbnail_url=f"https://storage.example.com/users/{user_id}/tracks/track-{i}-thumbnail.jpg"
            )
            for i in range(1, 6)  # Simulate 5 tracks per page
        ]
        
        return TracksListResponse(tracks=tracks, pagination=pagination)
        
    except Exception as e:
        print(f"Error listing user tracks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tracks: {str(e)}"
        )

@router.get("/{track_id}", response_model=TrackDetail)
async def get_track_detail(
    track_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific track.
    """
    try:
        user_id = current_user["uid"]
        
        # In a real implementation, you would query your database for this track
        # Here, we'll simulate the response
        
        # Check if track exists and belongs to user (simulated)
        if track_id not in ["track-1", "track-2", "track-3", "track-4", "track-5"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Track with ID {track_id} not found or does not belong to you"
            )
        
        # Simulated track detail
        track_detail = TrackDetail(
            track_id=track_id,
            title="Summer Beat",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            type="full_track",
            duration=180.5,
            audio_url=f"https://storage.example.com/users/{user_id}/tracks/{track_id}.mp3",
            waveform_data=[0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.4, 0.2, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.4, 0.2],
            thumbnail_url=f"https://storage.example.com/users/{user_id}/tracks/{track_id}-thumbnail.jpg",
            generation_parameters=GenerationParameters(
                prompt="Create an upbeat summer dance track with tropical vibes",
                vocal_settings="with_vocals",
                lyrics_settings="generate_lyrics",
                instruments=["synth", "drums", "bass", "guitar"],
                styles_themes=["summer", "dance", "tropical"],
                tempo=120,
                pitch="C Major"
            ),
            components=[
                TrackComponent(
                    component_type="vocals",
                    url=f"https://storage.example.com/users/{user_id}/tracks/{track_id}/vocals.mp3",
                    format="mp3"
                ),
                TrackComponent(
                    component_type="instrumental",
                    url=f"https://storage.example.com/users/{user_id}/tracks/{track_id}/instrumental.mp3",
                    format="mp3"
                ),
                TrackComponent(
                    component_type="lyrics",
                    url=f"https://storage.example.com/users/{user_id}/tracks/{track_id}/lyrics.txt",
                    format="txt"
                )
            ],
            metadata={
                "generated_by": "AI Music Generator v1.0",
                "tags": ["summer", "dance", "tropical"],
                "bpm": 120,
                "key": "C Major",
                "vocals_language": "English"
            }
        )
        
        return track_detail
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving track details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve track details: {str(e)}"
        )

@router.delete("/{track_id}", response_model=DeleteResponse)
async def delete_track(
    track_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a track from the user's library.
    """
    try:
        user_id = current_user["uid"]
        
        # In a real implementation, you would check if the track exists and belongs to the user
        # Here, we'll simulate the check
        if track_id not in ["track-1", "track-2", "track-3", "track-4", "track-5"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Track with ID {track_id} not found or does not belong to you"
            )
        
        # In a real implementation, you would delete the track from storage and database
        # storage_config.delete_folder(f"users/{user_id}/tracks/{track_id}")
        # db.tracks.delete_one({"track_id": track_id, "user_id": user_id})
        
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

@router.get("/{track_id}/download")
async def download_track(
    track_id: str,
    format: AudioFormat = Query(AudioFormat.MP3, description="Desired file format"),
    current_user: dict = Depends(get_current_user)
):
    """
    Download the track file in the specified format.
    """
    try:
        user_id = current_user["uid"]
        
        # In a real implementation, you would check if the track exists and belongs to the user
        # Here, we'll simulate the check
        if track_id not in ["track-1", "track-2", "track-3", "track-4", "track-5"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Track with ID {track_id} not found or does not belong to you"
            )
        
        # In a real implementation, you would get the file from storage
        # If the requested format is different from the stored format, you would convert it
        
        # Simulate file path (in a real implementation, this would point to an actual file)
        # file_path = f"temp/{user_id}/{track_id}.{format}"
        
        # For demonstration purposes, we'll return a simulated file response
        # In a real implementation, you would use FileResponse with an actual file path
        # return FileResponse(
        #     path=file_path,
        #     filename=f"track-{track_id}.{format}",
        #     media_type=f"audio/{format}"
        # )
        
        # Simulated response for this example
        headers = {
            "Content-Disposition": f'attachment; filename="track-{track_id}.{format}"'
        }
        return Response(
            content=b"Simulated file content",
            media_type=f"audio/{format}",
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error downloading track: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download track: {str(e)}"
        )