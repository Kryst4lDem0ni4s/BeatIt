import os
import shutil
from fastapi import APIRouter, Depends, BackgroundTasks, File, Form, HTTPException, Query, Response, UploadFile, WebSocket, status
import uuid
from datetime import datetime, timedelta
from .auth import get_current_user
from .files import GENERATION_STATUSES
from .lyrics import generate_lyrics_endpoint
from .tracks import get_track_info
from ..training.lyricsgen import LyricsGenerator
from ..training.vocalgen import generate_vocals
from ..training.instrumentalgen import InstrumentalGenerator
from ..training.musicgen import MusicGenerator
from config import storage_config
from ..models import model_types
from jobs import jobs_db, process_music_generation_job

router = APIRouter()

# @router.post("/generate-music", response_model=model_types.MusicGenerationResponse)
# async def generate_music(
#     request: model_types.MusicGenerationRequest,
#     current_user: dict = Depends(get_current_user)
# ):
#     try:
#         user_id = current_user["uid"]
        
#         # Generate a unique track ID
#         track_id = str(uuid.uuid4())
        
#         # Process lyrics based on settings
#         lyrics = None
#         if request.lyrics_settings == "generate lyrics":
#             lyrics = LyricsGenerator.generate_lyrics(
#                 prompt=request.text_prompt,
#                 reference_lyrics=request.style_references.lyrics_references if request.style_references else None,
#                 reference_mode=request.reference_usage_mode
#             )
#         elif request.lyrics_settings == "custom lyrics" and request.custom_lyrics:
#             lyrics = request.custom_lyrics
        
#         # Process vocals based on settings
#         vocal_track_path = None
#         if request.vocal_settings in ["with vocals", "only vocals"]:
#             if request.vocals_source == "generate vocals" and lyrics:
#                 vocal_track_path = generate_vocals(
#                     lyrics=lyrics,
#                     pitch=request.pitch,
#                     reference_tracks=request.style_references.vocals_references if request.style_references else None,
#                     reference_mode=request.reference_usage_mode,
#                     user_id=user_id
#                 )
#             # Handle custom vocals input case (would need file upload handling)
        
#         # Process instrumental based on settings
#         instrumental_track_path = None
#         if request.vocal_settings in ["with vocals", "no vocals"]:
#             if request.instrumental_source == "generate track":
#                 instrumental_track_path = InstrumentalGenerator.create_midi(
#                     prompt=request.text_prompt,
#                     tempo=request.tempo,
#                     desired_instruments=request.instruments,
#                     avoided_instruments=request.instruments_to_avoid,
#                     styles=request.styles_themes,
#                     avoided_styles=request.styles_themes_to_avoid,
#                     reference_tracks=request.style_references.music_references if request.style_references else None,
#                     reference_mode=request.reference_usage_mode,
#                     user_id=user_id
#                 )
#             # Handle custom instrumental input case (would need file upload handling)
        
#         # Combine tracks as needed
#         final_track_path = None
#         if vocal_track_path and instrumental_track_path:
#             final_track_path = MusicGenerator.combine_tracks(
#                 vocals_path=vocal_track_path,
#                 instrumental_path=instrumental_track_path,
#                 user_id=user_id,
#                 track_id=track_id
#             )
#         elif vocal_track_path:
#             final_track_path = vocal_track_path
#         elif instrumental_track_path:
#             final_track_path = instrumental_track_path
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No audio could be generated with the provided settings"
#             )
        
#         # Generate public URL for the track
#         storage_path = f"users/{user_id}/tracks/{track_id}/final.mp3"
#         audio_url = storage_config.upload_file(
#             local_path=final_track_path,
#             storage_path=storage_path
#         )
        
#         # Build metadata
#         metadata = {
#             "created_at": datetime.now().isoformat(),
#             "user_id": user_id,
#             "prompt": request.text_prompt,
#             "lyrics_mode": request.lyrics_settings,
#             "vocal_mode": request.vocal_settings,
#             "has_lyrics": request.lyrics_settings != "no lyrics",
#             "has_vocals": request.vocal_settings != "no vocals",
#             "has_instrumental": request.vocal_settings != "only vocals",
#             "tempo": request.tempo,
#             "pitch": request.pitch,
#             "styles_themes": request.styles_themes,
#             "instruments": request.instruments
#         }
        
#         # Store metadata in database
#         # db.tracks.insert_one({...}) # Uncomment and implement based on your DB
        
#         return model_types.MusicGenerationResponse(
#             status="success",
#             message="Music generated successfully",
#             track_id=track_id,
#             audio_url=audio_url,
#             metadata=metadata
#         )
    
#     except Exception as e:
#         # Log the error
#         print(f"Error in music generation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to generate music: {str(e)}"
#         )

@router.post("/generate-music", response_model=model_types.MusicGenerationResponse)
async def generate_music(
    request: model_types.MusicGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Orchestrator endpoint for the step-by-step music generation process.
    
    This endpoint coordinates the entire generation workflow by sequentially
    calling each step endpoint and managing the overall process flow.
    """
    try:
        user_id = current_user["uid"]
        
        # Step 1: Initialize session with basic parameters
        step1_request = model_types.Step1Request(
            text_prompt=request.text_prompt,
            vocal_settings=request.vocal_settings,
            lyrics_settings=request.lyrics_settings,
            custom_lyrics=request.custom_lyrics
        )
        
        # Call step1 to initialize session and handle lyrics generation if needed
        step1_response = await step1_basic_info(step1_request, current_user)
        session_id = step1_response.session_id
        
        # Step 2: Submit style references
        step2_request = model_types.Step2Request(
            style_references=request.style_references,
            reference_usage_mode=request.reference_usage_mode or "guidance_only"
        )
        
        # Call step2 to process style references
        step2_response = await step2_style_references(session_id, step2_request, current_user)
        
        # Step 3: Submit musical attributes
        step3_request = model_types.Step3Request(
            instruments=request.instruments or [],
            instruments_to_avoid=request.instruments_to_avoid or [],
            styles_themes=request.styles_themes or [],
            styles_themes_to_avoid=request.styles_themes_to_avoid or [],
            pitch=request.pitch,
            tempo=request.tempo
        )
        
        # Call step3 to process musical attributes
        step3_response = await step3_musical_attributes(session_id, step3_request, current_user)
        
        # Retrieve the updated session data
        session_data = generation_sessions[session_id]
        
        # Step 4: Review lyrics if needed
        if session_data["lyrics_settings"] != "no_lyrics":
            # For lyrics review/modification
            step4_request = model_types.Step4Request(
                regenerate_lyrics=False,  # Don't regenerate by default
                lyrics=None  # Use existing lyrics
            )
            
            # Call step4 to review/process lyrics
            await step4_lyrics_review(session_id, step4_request, current_user)
        
        # Step 5: Finalize and start generation
        step5_response = await step5_finalize(session_id, background_tasks, current_user)
        
        # Calculate estimated generation time
        estimated_time = calculate_estimated_time(request)
        
        # Return response with job information
        return model_types.MusicGenerationResponse(
            status="processing",
            message="Music generation started successfully",
            track_id=step5_response.job_id,
            audio_url=None,  # Will be available when generation completes
            estimated_time=estimated_time,
            metadata={
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "user_id": user_id,
                "prompt": request.text_prompt,
                "lyrics_mode": request.lyrics_settings,
                "vocal_mode": request.vocal_settings,
                "has_style_references": request.style_references is not None,
                "instruments_count": len(request.instruments or []),
                "styles_count": len(request.styles_themes or [])
            }
        )
        
    except Exception as e:
        print(f"Error in music generation orchestrator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to orchestrate music generation: {str(e)}"
        )

# Helper function to estimate generation time
def calculate_estimated_time(request: model_types.MusicGenerationRequest) -> int:
    """Calculate estimated processing time based on request complexity."""
    base_time = 60  # Base processing time in seconds
    
    # Add time for vocals processing
    if request.vocal_settings in ["with_vocals", "only_vocals"]:
        base_time += 60
        
    # Add time for each reference file
    if request.style_references:
        ref_count = 0
        if hasattr(request.style_references, "music_references") and request.style_references.music_references:
            ref_count += len(request.style_references.music_references)
        if hasattr(request.style_references, "vocals_references") and request.style_references.vocals_references:
            ref_count += len(request.style_references.vocals_references)
        if hasattr(request.style_references, "lyrics_references") and request.style_references.lyrics_references:
            ref_count += len(request.style_references.lyrics_references)
            
        base_time += ref_count * 15  # 15 seconds per reference
    
    # Add time for complex prompts
    if request.text_prompt and len(request.text_prompt) > 200:
        base_time += 30
        
    return base_time


# Response model
@router.get("/generation-status/{track_id}", response_model=model_types.GenerationStatusResponse, 
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

def get_generation_status_from_task_manager(track_id: str) -> model_types.GenerationStatusResponse:
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
        return model_types.GenerationStatusResponse(
            status=GENERATION_STATUSES["QUEUED"],
            message="Your music generation request is in the queue and will be processed soon",
            estimated_time=120  # 2 minutes
        )
    elif status_index == 1:
        return model_types.GenerationStatusResponse(
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
        return model_types.GenerationStatusResponse(
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
        return model_types.GenerationStatusResponse(
            status=GENERATION_STATUSES["FAILED"],
            message="Music generation failed due to an error",
            details={
                "error": "Model inference failed",
                "error_details": "The AI model encountered an error with your specific prompt, please try again with different parameters"
            }
        )


@router.post("/generate/init", response_model=model_types.MusicGenerationInitResponse)
async def init_music_generation(
    request: model_types.MusicGenerationInitRequest,
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
        
        return model_types.MusicGenerationInitResponse(
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

# @staticmethod
# def generate_waveform_data(audio_path, num_points=100):
#     """
#     Generate waveform data for visualization from an audio file.
    
#     Args:
#         audio_path (str): Path to the audio file
#         num_points (int): Number of data points to generate
        
#     Returns:
#         List[float]: Array of amplitude values for visualization
#     """
#     try:
#         import numpy as np
#         from pydub import AudioSegment
        
#         # Load audio file
#         audio = AudioSegment.from_file(audio_path)
        
#         # Convert to numpy array (mono)
#         samples = np.array(audio.get_array_of_samples())
#         if audio.channels == 2:
#             samples = samples.reshape((-1, 2)).mean(axis=1)
        
#         # Normalize
#         samples = samples / np.max(np.abs(samples))
        
#         # Resample to desired number of points
#         samples_length = len(samples)
#         points_per_sample = samples_length // num_points
        
#         waveform_data = []
#         for i in range(num_points):
#             start = i * points_per_sample
#             end = min(start + points_per_sample, samples_length)
#             if start < samples_length:
#                 # Use absolute max value in this segment
#                 waveform_data.append(float(np.max(np.abs(samples[start:end]))))
        
#         return waveform_data
#     except Exception as e:
#         print(f"Error generating waveform data: {str(e)}")
#         # Return a flat waveform if there's an error
#         return [0.5] * num_points
    
# Session storage (replace with database in production)
generation_sessions = {}

# Request/Response models for Step 1
@router.post("/step1", response_model=model_types.Step1Response)
async def step1_basic_info(
    request: model_types.Step1Request,
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
        
        # Generate lyrics if requested by calling the lyrics endpoint
        if request.lyrics_settings == "generate_lyrics":
            try:
                # Prepare lyrics generation request
                lyrics_request = model_types.LyricsGenerationRequest(
                    prompt=request.text_prompt,
                    style=None,  # Use default style
                    reference_lyrics=None  # No references at this stage
                )
                
                # Call lyrics generation endpoint
                lyrics_response = await generate_lyrics_endpoint(lyrics_request, current_user)
                
                # Store generated lyrics in session
                session_data["generated_lyrics"] = lyrics_response.lyrics_text
                session_data["lyrics_id"] = lyrics_response.lyrics_id
                
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
        
        return model_types.Step1Response(
            session_id=session_id,
            next_step=next_step
        )
        
    except Exception as e:
        print(f"Error in step 1: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step 1: {str(e)}"
        )

@router.post("/step2/{session_id}", response_model=model_types.StepResponse)
async def step2_style_references(
    session_id: str,
    request: model_types.Step2Request,
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
        
        # Add helper function
        def file_exists(file_id: str, user_id: str) -> bool:
            """Check if a file exists and belongs to the user."""
            # In a real implementation, query your database
            # For now, we'll just check if the file_id is not empty
            return bool(file_id)

        # In step2 endpoint
        if request.style_references:
            for ref_id in request.style_references.music_references or []:
                if not file_exists(ref_id, user_id):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Reference file {ref_id} not found or doesn't belong to you"
                    )
        
        # Define next step information
        next_step = {
            "step": 3,
            "endpoint": f"/api/music/step3/{session_id}",
            "description": "Submit musical attributes",
            "required_fields": ["instruments", "styles_themes", "pitch", "tempo"]
        }
        
        return model_types.StepResponse(
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

@router.post("/step3/{session_id}", response_model=model_types.StepResponse)
async def step3_musical_attributes(
    session_id: str,
    request: model_types.Step3Request,
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
        
        return model_types.StepResponse(
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

@router.post("/step4/{session_id}", response_model=model_types.Step4Response)
async def step4_lyrics_review(
    session_id: str,
    request: model_types.Step4Request,
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
        
        # Handle case where no lyrics are needed but step4 is called anyway
        if session_data["lyrics_settings"] == "no_lyrics":
            # Skip to step 5 instead of raising an error
            session_data["step"] = 4  # Still increment step
            
            next_step = {
                "step": 5,
                "endpoint": f"/api/music/step5/{session_id}/finalize",
                "description": "Finalize all parameters and start generation",
                "required_fields": []
            }
            
            return model_types.Step4Response(
                session_id=session_id,
                lyrics=None,  # No lyrics needed
                session_data=session_data,
                next_step=next_step
            )
        
        # Update session data
        session_data["step"] = 4
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Get current lyrics
        lyrics = session_data.get("generated_lyrics", "")
        
        # Handle lyrics regeneration if requested
        if request.regenerate_lyrics:
            try:
                # Create lyrics generation request
                lyrics_request = model_types.LyricsGenerationRequest(
                    prompt=session_data["text_prompt"],
                    style=None,  # Default style
                    reference_lyrics=session_data.get("style_references", {}).get("lyrics_references")
                )
                
                # Call lyrics generation endpoint
                lyrics_response = await generate_lyrics_endpoint(lyrics_request, current_user)
                
                # Update session with new lyrics
                lyrics = lyrics_response.lyrics_text
                session_data["generated_lyrics"] = lyrics
                session_data["lyrics_id"] = lyrics_response.lyrics_id
                session_data["lyrics_regenerated"] = True
                
            except Exception as e:
                print(f"Error regenerating lyrics: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to regenerate lyrics: {str(e)}"
                )
                
        elif request.lyrics is not None:
            # Use modified lyrics provided by user
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
        
        return model_types.Step4Response(
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


@router.post("/step5/{session_id}/finalize", response_model=model_types.Step5Response)
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
        
        # Start the actual generation process in the background
        background_tasks.add_task(
            process_music_generation_job,
            job_id=job_id,
            user_id=user_id,
            parameters=job_params
        )
        
        # Optionally, add session cleanup to prevent memory leaks
        # This would be after a timeout period in a real application
        # background_tasks.add_task(
        #     cleanup_session_after_timeout,
        #     session_id=session_id,
        #     timeout=3600  # 1 hour
        # )
        
        return model_types.Step5Response(
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


        
@router.get("/models")
async def get_available_models(current_user: dict = Depends(get_current_user)):
    """
    Get information about available ML models for music generation.
    """
    models = [
        {
            "id": "musicgen",
            "name": "MusicGen",
            "provider": "Meta",
            "description": "Text-to-music model that generates high-quality music from text descriptions",
            "capabilities": ["text_to_music", "melody_continuation"],
            "specialties": ["diverse_genres", "high_fidelity_output"],
            "max_duration": 30,  # in seconds
            "supported_formats": ["mp3", "wav"]
        },
        {
            "id": "musiclm",
            "name": "MusicLM",
            "provider": "Google",
            "description": "Model that creates music based on textual descriptions",
            "capabilities": ["text_to_music"],
            "specialties": ["long_form_generation", "style_consistency"],
            "max_duration": 60,  # in seconds
            "supported_formats": ["mp3", "wav"]
        },
        {
            "id": "suno",
            "name": "Suno AI",
            "provider": "Suno",
            "description": "Comprehensive music generation tool with vocals and lyrics",
            "capabilities": ["text_to_music", "lyric_generation", "vocal_synthesis"],
            "specialties": ["complete_song_creation", "high_quality_vocals"],
            "max_duration": 180,  # in seconds
            "supported_formats": ["mp3", "wav", "stems"]
        },
        {
            "id": "stable_audio",
            "name": "Stable Audio 2.0",
            "provider": "Stability AI",
            "description": "Advanced audio generation with extensive customization",
            "capabilities": ["text_to_music", "audio_to_audio", "style_transfer"],
            "specialties": ["sound_effects", "long_form_generation", "high_fidelity"],
            "max_duration": 180,  # in seconds
            "supported_formats": ["mp3", "wav", "flac"]
        }
    ]
    
    return models

@router.get("/instruments")
async def get_instrument_options(current_user: dict = Depends(get_current_user)):
    """
    Get list of supported instruments for music generation.
    """
    instruments = {
        "strings": [
            {"id": "violin", "name": "Violin", "family": "strings", "range": "high"},
            {"id": "viola", "name": "Viola", "family": "strings", "range": "mid-high"},
            {"id": "cello", "name": "Cello", "family": "strings", "range": "mid-low"},
            {"id": "double_bass", "name": "Double Bass", "family": "strings", "range": "low"},
            {"id": "acoustic_guitar", "name": "Acoustic Guitar", "family": "strings", "range": "mid"},
            {"id": "electric_guitar", "name": "Electric Guitar", "family": "strings", "range": "mid"},
            {"id": "bass_guitar", "name": "Bass Guitar", "family": "strings", "range": "low"},
            {"id": "harp", "name": "Harp", "family": "strings", "range": "wide"}
        ],
        "woodwinds": [
            {"id": "flute", "name": "Flute", "family": "woodwinds", "range": "high"},
            {"id": "clarinet", "name": "Clarinet", "family": "woodwinds", "range": "mid-high"},
            {"id": "oboe", "name": "Oboe", "family": "woodwinds", "range": "mid-high"},
            {"id": "bassoon", "name": "Bassoon", "family": "woodwinds", "range": "low"},
            {"id": "saxophone", "name": "Saxophone", "family": "woodwinds", "range": "mid"}
        ],
        "brass": [
            {"id": "trumpet", "name": "Trumpet", "family": "brass", "range": "high"},
            {"id": "trombone", "name": "Trombone", "family": "brass", "range": "mid-low"},
            {"id": "french_horn", "name": "French Horn", "family": "brass", "range": "mid"},
            {"id": "tuba", "name": "Tuba", "family": "brass", "range": "low"}
        ],
        "percussion": [
            {"id": "drums", "name": "Drum Kit", "family": "percussion", "range": "varied"},
            {"id": "timpani", "name": "Timpani", "family": "percussion", "range": "low"},
            {"id": "xylophone", "name": "Xylophone", "family": "percussion", "range": "high"},
            {"id": "marimba", "name": "Marimba", "family": "percussion", "range": "mid"},
            {"id": "cymbals", "name": "Cymbals", "family": "percussion", "range": "high"}
        ],
        "keyboards": [
            {"id": "piano", "name": "Piano", "family": "keyboards", "range": "wide"},
            {"id": "organ", "name": "Organ", "family": "keyboards", "range": "wide"},
            {"id": "harpsichord", "name": "Harpsichord", "family": "keyboards", "range": "mid-high"},
            {"id": "synthesizer", "name": "Synthesizer", "family": "keyboards", "range": "wide"}
        ],
        "electronic": [
            {"id": "synth_lead", "name": "Synth Lead", "family": "electronic", "range": "high"},
            {"id": "synth_pad", "name": "Synth Pad", "family": "electronic", "range": "mid"},
            {"id": "synth_bass", "name": "Synth Bass", "family": "electronic", "range": "low"},
            {"id": "drum_machine", "name": "Drum Machine", "family": "electronic", "range": "varied"},
            {"id": "sampler", "name": "Sampler", "family": "electronic", "range": "varied"}
        ]
    }
    
    return instruments

@router.get("/styles")
async def get_style_options(current_user: dict = Depends(get_current_user)):
    """
    Get list of supported musical styles and themes.
    """
    styles = {
        "genres": [
            {"id": "rock", "name": "Rock", "description": "Guitar-driven music with strong beats and often bold lyrics"},
            {"id": "pop", "name": "Pop", "description": "Contemporary mainstream music with catchy melodies"},
            {"id": "jazz", "name": "Jazz", "description": "Complex harmonies with improvisation and syncopated rhythms"},
            {"id": "classical", "name": "Classical", "description": "Traditional Western music from the 18th-19th century"},
            {"id": "electronic", "name": "Electronic", "description": "Music produced with electronic instruments and technology"},
            {"id": "hip_hop", "name": "Hip Hop", "description": "Rhythmic music with rapping and beats"},
            {"id": "r_and_b", "name": "R&B", "description": "Rhythm and blues with soulful singing and grooves"},
            {"id": "country", "name": "Country", "description": "Music with roots in Southern US folk music"},
            {"id": "folk", "name": "Folk", "description": "Traditional music passed through generations"},
            {"id": "metal", "name": "Metal", "description": "Heavy and aggressive rock music"}
        ],
        "sub_genres": [
            {"id": "indie_rock", "name": "Indie Rock", "parent": "rock", "description": "Rock music produced independently from commercial record labels"},
            {"id": "hard_rock", "name": "Hard Rock", "parent": "rock", "description": "Aggressive rock with heavy guitars and drums"},
            {"id": "synth_pop", "name": "Synth Pop", "parent": "pop", "description": "Pop music with prominent synthesizer sounds"},
            {"id": "baroque", "name": "Baroque", "parent": "classical", "description": "Classical music from 1600-1750 with ornate structures"},
            {"id": "techno", "name": "Techno", "parent": "electronic", "description": "Electronic dance music with regular beats and synthetic sounds"}
        ],
        "moods": [
            {"id": "happy", "name": "Happy", "description": "Uplifting and joyful"},
            {"id": "sad", "name": "Sad", "description": "Somber and melancholic"},
            {"id": "energetic", "name": "Energetic", "description": "High energy and lively"},
            {"id": "relaxed", "name": "Relaxed", "description": "Calm and peaceful"},
            {"id": "aggressive", "name": "Aggressive", "description": "Intense and forceful"},
            {"id": "romantic", "name": "Romantic", "description": "Passionate and tender"}
        ],
        "eras": [
            {"id": "50s", "name": "1950s", "description": "Early rock and roll, doo-wop"},
            {"id": "60s", "name": "1960s", "description": "British invasion, psychedelic rock, Motown"},
            {"id": "70s", "name": "1970s", "description": "Disco, progressive rock, punk"},
            {"id": "80s", "name": "1980s", "description": "New wave, synth-pop, early hip hop"},
            {"id": "90s", "name": "1990s", "description": "Grunge, boy bands, gangsta rap"},
            {"id": "2000s", "name": "2000s", "description": "Nu-metal, emo, crunk"}
        ],
        "themes": [
            {"id": "nature", "name": "Nature", "description": "Inspired by natural environments and phenomena"},
            {"id": "urban", "name": "Urban", "description": "City life and metropolitan themes"},
            {"id": "space", "name": "Space", "description": "Cosmic and outer space themes"},
            {"id": "fantasy", "name": "Fantasy", "description": "Mythical and magical themes"},
            {"id": "futuristic", "name": "Futuristic", "description": "Forward-looking technological themes"},
            {"id": "retro", "name": "Retro", "description": "Nostalgic throwbacks to earlier styles"}
        ]
    }
    
    return styles
