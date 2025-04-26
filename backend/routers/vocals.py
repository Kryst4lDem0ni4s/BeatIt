from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
from fastapi import BackgroundTasks, Depends, HTTPException, APIRouter, Query, Response, status
from ..config import StorageConfig
from ..models import model_types
from .auth import get_current_user
from vocalgen import generate_vocals

router = APIRouter()        

vocals_db = {}

# Request and Response Models for Vocals Generation

@router.post("/generate", response_model=model_types.VocalsGenerationResponse)
async def generate_vocals_endpoint(
    request: model_types.VocalsGenerationRequest,
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
        
        return model_types.VocalsGenerationResponse(
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

@router.put("/{vocals_id}/customize", response_model=model_types.VocalsCustomizationResponse)
async def customize_vocals_endpoint(
    vocals_id: str,
    request: model_types.VocalsCustomizationRequest,
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
        
        return model_types.VocalsCustomizationResponse(
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
        vocals_url = StorageConfig.upload_file(
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
        vocal_track_path = model_types.customize_vocals(
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
        vocals_url = StorageConfig.upload_file(
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
def calculate_estimated_time(lyrics_text: str, style: Optional[model_types.VocalStylePreferences], pitch: Optional[model_types.PitchAdjustments]) -> int:
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
@router.get("/{vocals_id}", response_model=model_types.VocalsCustomizationResponse)
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
        
        return model_types.VocalsCustomizationResponse(
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
        