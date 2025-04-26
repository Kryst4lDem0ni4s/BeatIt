from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
from fastapi import BackgroundTasks, Depends, HTTPException, APIRouter, Query, Response
from grpc import Status
from backend.config import StorageConfig
from backend.models import model_types
from backend.routers.auth import get_current_user
from instrumentalgen import InstrumentalGenerator
from vocalgen import generate_vocals

router = APIRouter()    

# In-memory storage for instrumental tracks (replace with database in production)
instrumentals_db = {}

# Request and Response Models for Instrumental Generation

@router.post("/generate", response_model=model_types.InstrumentalGenerationResponse)
async def generate_instrumental(
    request: model_types.InstrumentalGenerationRequest,
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
        
        return model_types.InstrumentalGenerationResponse(
            instrumental_id=instrumental_id,
            status="processing",
            estimated_completion_time=estimated_time,
            metadata=instrumental_data["metadata"]
        )
        
    except Exception as e:
        print(f"Error in instrumental generation: {str(e)}")
        raise HTTPException(
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process instrumental generation: {str(e)}"
        )

@router.put("/{instrumental_id}/customize", response_model=model_types.InstrumentalCustomizationResponse)
async def customize_instrumental(
    instrumental_id: str,
    request: model_types.InstrumentalCustomizationRequest,
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
                status_code=Status.HTTP_404_NOT_FOUND,
                detail=f"Instrumental with ID {instrumental_id} not found"
            )
        
        # Check if instrumental belongs to user
        instrumental_data = instrumentals_db[instrumental_id]
        if instrumental_data["user_id"] != user_id:
            raise HTTPException(
                status_code=Status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to customize this instrumental"
            )
        
        # Check if instrumental is in a customizable state
        if instrumental_data["status"] == "processing":
            raise HTTPException(
                status_code=Status.HTTP_400_BAD_REQUEST,
                detail="Instrumental is still processing and cannot be customized yet"
            )
        
        if instrumental_data["status"] == "failed":
            raise HTTPException(
                status_code=Status.HTTP_400_BAD_REQUEST,
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
        
        return model_types.InstrumentalCustomizationResponse(
            instrumental_id=instrumental_id,
            status="processing",
            metadata=instrumental_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in instrumental customization: {str(e)}")
        raise HTTPException(
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
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
        instrumental_url = StorageConfig.upload_file(
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
    instrument_adjustments: List[model_types.InstrumentAdjustment], 
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
        instrumental_url = StorageConfig.upload_file(
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
def calculate_estimated_time(request: model_types.InstrumentalGenerationRequest) -> int:
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
@router.get("/{instrumental_id}", response_model=model_types.InstrumentalCustomizationResponse)
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
                status_code=Status.HTTP_404_NOT_FOUND,
                detail=f"Instrumental with ID {instrumental_id} not found"
            )
        
        # Check if instrumental belongs to user
        instrumental_data = instrumentals_db[instrumental_id]
        if instrumental_data["user_id"] != user_id:
            raise HTTPException(
                status_code=Status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this instrumental"
            )
        
        return model_types.InstrumentalCustomizationResponse(
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
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve instrumental status: {str(e)}"
        )