from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum

# class SignUpRequest(BaseModel):
#     email: str
#     username: str
#     password: str
#     phonenumber: str = Field(..., pattern=r'^\+91\d{10}$')

# Define valid audio file types and file size limits
VALID_AUDIO_TYPES = ["vocals", "instrumental", "reference"]
VALID_AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
VALID_FILE_TYPES = ["vocals", "instrumental", "reference", "lyrics"]
VALID_FILE_PURPOSES = ["direct_use", "inspiration"]
VALID_AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
VALID_LYRICS_EXTENSIONS = [".txt", ".md"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Define missing enumerations
class GenTypeEnum(str, Enum):
    INSTRUMENTAL_VOCAL = "instrumental_vocal"
    INSTRUMENTAL_ONLY = "instrumental_only"
    VOCAL_ONLY = "vocal_only"

class VocalSettingsEnum(str, Enum):
    WITH_VOCALS = "with_vocals"
    NO_VOCALS = "no_vocals"
    ONLY_VOCALS = "only_vocals"

class VocalsSourceEnum(str, Enum):
    GENERATE_VOCALS = "generate_vocals"
    CUSTOM_INPUT = "custom_input"

class InstrumentalSourceEnum(str, Enum):
    GENERATE_TRACK = "generate_track"
    CUSTOM_INPUT = "custom_input"

class LyricsSettingsEnum(str, Enum):
    GENERATE_LYRICS = "generate_lyrics"
    CUSTOM_LYRICS = "custom_lyrics"
    NO_LYRICS = "no_lyrics"

class ReferenceUsageModeEnum(str, Enum):
    GUIDANCE_ONLY = "guidance_only"
    DIRECT_USAGE = "direct_usage"

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
class LoginRequest(BaseModel):
    email: str
    password: str
    
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
    
class JobStatusResponse(BaseModel):
    status: str
    progress: float
    estimated_time: Optional[int] = None  # in seconds
    message: Optional[str] = None
    
class StyleReferences(BaseModel):
    sample_lyrics: Optional[List[str]] = None
    sample_music_tracks: Optional[List[str]] = None
    lyrics_for_usage: Optional[List[str]] = None
    music_tracks_for_usage: Optional[List[str]] = None
    music_tracks_for_sampling: Optional[List[str]] = None
    vocal_tracks_for_usage: Optional[List[str]] = None
    vocal_tracks_for_sampling: Optional[List[str]] = None

class Step1Request(BaseModel):
    text_prompt: str
    vocal_settings: Literal["no_vocals", "only_vocals", "with_vocals"]
    lyrics_settings: Literal["no_lyrics", "custom_lyrics", "generate_lyrics"]
    custom_lyrics: Optional[str] = None
    lyrics_style: Optional[Dict[str, Any]] = None
    vocals_source: Optional[Literal["generate_vocals", "custom_vocals"]] = None
    instrumental_source: Optional[Literal["generate_music", "custom_music"]] = None
    style_references: Optional[StyleReferences] = None

class Step1Response(BaseModel):
    session_id: str
    session_data: Dict[str, Any]
    next_step: Dict[str, Any]

# Request/Response models for Step 2
class StyleReferences(BaseModel):
    lyrics_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for lyrics references")
    vocals_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for vocal references") 
    music_references: Optional[List[str]] = Field(default=[], description="Array of file_ids for music references")

class Step2Request(BaseModel):
    # Style references
    style_references: Optional[StyleReferences] = None
    reference_usage_mode: Optional[Literal["guidance_only", "direct_use", "hybrid"]] = "guidance_only"
    
    # Instrument selection
    instruments: Optional[List[str]] = None
    instruments_to_avoid: Optional[List[str]] = None
    
    # Style preferences
    styles_themes: Optional[List[str]] = None
    styles_themes_to_avoid: Optional[List[str]] = None
    
    # Musical attributes
    pitch: Optional[str] = None
    tempo: Optional[int] = None
    
    # Voice type (for vocals)
    voice_type: Optional[Literal["male", "female", "neutral"]] = "neutral"
    
    # Track duration in seconds (max 300 seconds = 5 minutes)
    duration: Optional[int] = 180  # Default 3 minutes


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


class GenerationStatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    estimated_time: Optional[int] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    
# class EmailRequest(BaseModel):
#     email: str
    
# class UpdatePasswordRequest(BaseModel):
#     uid: str
#     new_password: str

# class Profile(BaseModel):
#     fullname: str
#     password: constr(min_length=8)
#     phonenumber: str = Field(..., pattern=r'^\+91\d{10}$')
#     address: str
#     email: str
