import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from auth import get_current_user
from backend.routers.files import GENERATION_STATUSES
from backend.routers.tracks import get_track_info
from ..training.lyricsgen import LyricsGenerator
from ..training.vocalgen import generate_vocals
from ..training.instrumentalgen import InstrumentalGenerator
from ..training.musicgen import MusicGenerator
from config import storage_config
from ..models import model_types

router = APIRouter()

# In-memory job storage (replace with a database in production)
jobs_db = {}

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
                
            lyrics = LyricsGenerator.generate_lyrics(
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
        

@router.get("/jobs/{job_id}/status", response_model=model_types.JobStatusResponse)
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
        
        return model_types.JobStatusResponse(
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

@router.get("/jobs/{job_id}/result", response_model=model_types.GenerationResultResponse)
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
        return model_types.GenerationResultResponse(**job_data["result"])
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving job result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job result: {str(e)}"
        )