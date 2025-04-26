from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
from fastapi import BackgroundTasks, Depends, HTTPException, APIRouter, Query, Response, status
from backend.config import StorageConfig
from backend.models import model_types
from backend.routers.auth import get_current_user
from instrumentalgen import InstrumentalGenerator
from lyricsgen import LyricsGenerator
from vocalgen import generate_vocals

router = APIRouter() 

# In-memory lyrics storage (replace with database in production)
lyrics_db = {}

# Request and Response Models for Lyrics Generation
@router.post("/generate", response_model=model_types.LyricsGenerationResponse)
async def generate_lyrics_endpoint(
    request: model_types.LyricsGenerationRequest,
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
            lyrics_text = LyricsGenerator.generate_lyrics(
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
        
        return model_types.LyricsGenerationResponse(
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

@router.put("/{lyrics_id}/modify", response_model=model_types.LyricsModificationResponse)
async def modify_lyrics(
    lyrics_id: str,
    request: model_types.LyricsModificationRequest,
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
        
        return model_types.LyricsModificationResponse(
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
@router.get("/{lyrics_id}", response_model=model_types.LyricsGenerationResponse)
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
        
        return model_types.LyricsGenerationResponse(
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

 