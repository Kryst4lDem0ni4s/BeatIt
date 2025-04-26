import os
import shutil
from typing import Any, Dict, Optional
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from datetime import datetime
from .auth import get_current_user
from .tracks import get_track_info
from config import storage_config
from models import model_types

router = APIRouter()

# Response model
@router.post("/upload-audio", response_model=model_types.FileUploadResponse)
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
    if type not in model_types.VALID_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio type. Must be one of: {', '.join(model_types.VALID_AUDIO_TYPES)}"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in model_types.VALID_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Supported formats: {', '.join(model_types.VALID_AUDIO_EXTENSIONS)}"
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
                if file_size > model_types.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds the maximum allowed size of {model_types.MAX_FILE_SIZE // (1024 * 1024)} MB"
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
        
        return model_types.AudioUploadResponse(
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

# Response models

# 1.1 File Upload Endpoint
# @router.post("/upload", response_model=model_types.FileUploadResponse)
# async def upload_file(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     file_type: str = Form(...),
#     purpose: str = Form(...),
#     current_user: dict = Depends(get_current_user)
# ):
#     """
#     Upload an audio or lyrics file for music generation.
    
#     Parameters:
#     - file: The file to upload
#     - file_type: Type of file (vocals, instrumental, reference, lyrics)
#     - purpose: How the file will be used (direct_use, inspiration)
    
#     Returns:
#     - file_id: Unique identifier for the uploaded file
#     - file_url: URL to access the uploaded file
#     - status: Success/failure message
#     """
#     try:
#         # Get user ID from authenticated user
#         user_id = current_user["uid"]
        
#         # Validate file type
#         if file_type not in model_types.VALID_FILE_TYPES:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid file type. Must be one of: {', '.join(model_types.VALID_FILE_TYPES)}"
#             )
            
#         # Validate purpose
#         if purpose not in model_types.VALID_FILE_PURPOSES:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid purpose. Must be one of: {', '.join(model_types.VALID_FILE_PURPOSES)}"
#             )
        
#         # Get file extension and validate based on file type
#         file_extension = os.path.splitext(file.filename)[1].lower()
        
#         if file_type == "lyrics" and file_extension not in model_types.VALID_LYRICS_EXTENSIONS:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid lyrics file format. Supported formats: {', '.join(model_types.VALID_LYRICS_EXTENSIONS)}"
#             )
#         elif file_type != "lyrics" and file_extension not in model_types.VALID_AUDIO_EXTENSIONS:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid audio file format. Supported formats: {', '.join(model_types.VALID_AUDIO_EXTENSIONS)}"
#             )
        
#         # Generate a unique file ID
#         file_id = str(uuid.uuid4())
        
#         # Create temporary file to save upload
#         temp_dir = f"temp/{user_id}/{file_id}"
#         os.makedirs(temp_dir, exist_ok=True)
#         temp_file_path = os.path.join(temp_dir, f"original{file_extension}")
        
#         # Save uploaded file to temporary location
#         with open(temp_file_path, "wb") as buffer:
#             # Read file in chunks to handle large files efficiently
#             content = await file.read(1024 * 1024)  # Read 1MB at a time
#             file_size = 0
            
#             while content:
#                 file_size += len(content)
#                 if file_size > model_types.MAX_FILE_SIZE:
#                     # Clean up the temp dir if file is too large
#                     shutil.rmtree(temp_dir, ignore_errors=True)
#                     raise HTTPException(
#                         status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
#                         detail=f"File size exceeds the maximum allowed size of {model_types.MAX_FILE_SIZE // (1024 * 1024)} MB"
#                     )
#                 buffer.write(content)
#                 content = await file.read(1024 * 1024)
        
#         # Define storage path based on file type and user
#         storage_path = f"users/{user_id}/{file_type}/{file_id}{file_extension}"
        
#         # Upload file to configured storage
#         file_url = storage_config.upload_file(
#             local_path=temp_file_path,
#             storage_path=storage_path
#         )
        
#         # Store metadata in database
#         metadata = {
#             "file_id": file_id,
#             "file_name": file.filename,
#             "file_type": file_type,
#             "purpose": purpose,
#             "size": file_size,
#             "upload_date": datetime.now(),
#             "file_url": file_url,
#             "user_id": user_id,
#             "content_type": file.content_type,
#             "storage_path": storage_path
#         }
        
#         # In a real implementation, save this metadata to your database
#         # db.files.insert_one(metadata)
        
#         # Schedule temporary directory cleanup as background task
#         background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        
#         return model_types.FileUploadResponse(
#             file_id=file_id,
#             file_url=file_url,
#             status="success"
#         )
        
#     except HTTPException:
#         # Re-raise HTTP exceptions
#         raise
#     except Exception as e:
#         print(f"Error in file upload: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to upload file: {str(e)}"
#         )

# 1.2 File Retrieval Endpoint
@router.get("/files/{file_id}", response_model=model_types.FileMetadata)
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
        
        return model_types.FileMetadata(**metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving file metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve file metadata: {str(e)}"
        )

# 1.3 File Deletion Endpoint
@router.delete("/files/{file_id}", response_model=model_types.FileDeleteResponse)
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
        
        return model_types.FileDeleteResponse(
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