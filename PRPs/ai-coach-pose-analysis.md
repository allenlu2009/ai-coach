# AI Coach - Athlete Pose Analysis System

name: "AI Coach Video Pose Analysis System PRP"
description: |
  Comprehensive implementation plan for an AI coach system that analyzes athlete poses from uploaded videos and provides conversational feedback through a chat interface.

---

## Goal
Build a complete AI coach system that:
1. Accepts video uploads through a web interface
2. Processes videos using MediaPipe pose detection to extract 3D landmarks
3. Provides intelligent coaching feedback through an AI-powered chat interface
4. Leverages RTX 3060 GPU for real-time video processing and AI inference
5. Offers real-time pose visualization and analysis

## Why
- **Athletic Performance Enhancement**: Provides athletes with immediate, data-driven feedback on their form and technique
- **Accessibility**: Makes professional coaching insights available to athletes who may not have access to human coaches
- **Objective Analysis**: Uses computer vision to provide unbiased, consistent pose analysis
- **GPU Optimization**: Leverages existing RTX 3060 hardware for efficient real-time processing
- **Scalable Solution**: Web-based system can serve multiple athletes simultaneously

## What
A web application with three main components:
1. **Frontend**: React/HTML interface for video upload and chat interaction
2. **Backend API**: FastAPI server handling video processing and AI chat
3. **Pose Analysis Engine**: MediaPipe-based system for pose detection and analysis

### Success Criteria
- [ ] Users can upload video files through web interface
- [ ] System processes videos and extracts 33 3D pose landmarks per frame
- [ ] AI coach provides meaningful feedback based on pose analysis
- [ ] Real-time chat interface responds within 2 seconds
- [ ] System supports common video formats (MP4, AVI, MOV)
- [ ] Pose overlay visualization shows detected landmarks on video
- [ ] GPU acceleration reduces processing time by >50% vs CPU-only

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
  why: Official MediaPipe pose detection guide with 33 landmark specifications
  
- url: https://learnopencv.com/ai-fitness-trainer-using-mediapipe/
  why: Practical implementation patterns for AI fitness coaching systems
  
- url: https://towardsdatascience.com/human-pose-tracking-with-mediapipe-rerun-showcase-125053cfe64f
  why: 3D pose analysis techniques and best practices
  
- url: https://fastapi.tiangolo.com/tutorial/request-files/
  why: FastAPI file upload patterns and video processing
  
- file: examples/pose.py
  why: |
    Reference implementation showing:
    - MediaPipe pose detection setup
    - 3D landmark extraction (pose_world_landmarks)  
    - Video processing pipeline with OpenCV
    - GPU optimization patterns
    - Frame-by-frame analysis approach
    
- file: src/config/model_configs.py
  why: Configuration patterns for GPU memory management and model loading
  
- file: tests/conftest.py
  why: Testing framework patterns with fixtures and mocking approaches
```

### Current Codebase Tree
```bash
ai_coach/
├── examples/
│   └── pose.py                    # MediaPipe pose detection reference
├── src/
│   ├── config/
│   │   ├── model_configs.py      # GPU config patterns
│   │   └── __init__.py
│   └── perplexity/              # Existing ML pipeline structure
│       ├── models.py            # Pydantic model patterns
│       ├── cli.py               # CLI interface patterns
│       └── evaluator.py         # Processing pipeline patterns
├── tests/
│   ├── conftest.py              # Testing framework setup
│   └── test_*.py                # Test patterns
├── pyproject.toml               # uv dependency management
├── CLAUDE.md                    # Project guidelines
└── venv_linux/                  # Virtual environment
```

### Desired Codebase Tree with New Files
```bash
ai_coach/
├── src/
│   ├── ai_coach/                      # New AI coach module
│   │   ├── __init__.py
│   │   ├── models.py                  # Pydantic models for pose analysis
│   │   ├── pose_analyzer.py           # MediaPipe pose detection logic
│   │   ├── coach_agent.py             # AI coaching conversation logic
│   │   ├── video_processor.py         # Video upload and processing
│   │   └── api.py                     # FastAPI routes and endpoints
│   ├── frontend/                      # Web interface
│   │   ├── static/
│   │   │   ├── index.html            # Video upload and chat UI
│   │   │   ├── style.css             # UI styling
│   │   │   └── app.js                # Frontend JavaScript
│   │   └── templates/
├── tests/
│   ├── test_pose_analyzer.py         # Pose detection tests
│   ├── test_coach_agent.py           # AI coach tests
│   ├── test_video_processor.py       # Video processing tests
│   └── test_api.py                   # API endpoint tests
└── uploads/                          # Temporary video storage
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: MediaPipe requires specific setup for GPU acceleration
# Must configure model_complexity and enable_segmentation correctly
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2,          # Highest accuracy for coaching
    enable_segmentation=True     # Required for 3D landmarks
)

# CRITICAL: OpenCV VideoCapture with FastAPI UploadFile requires BytesIO
# Cannot pass file path directly - must write to temporary file first
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
temp_file.write(await video_file.read())
cap = cv2.VideoCapture(temp_file.name)

# CRITICAL: RTX 3060 CUDA memory management
# Must clear GPU cache between video processing sessions
torch.cuda.empty_cache()

# CRITICAL: MediaPipe coordinate systems
# Image coordinates (pixels) vs World coordinates (meters)
# Use pose_world_landmarks for 3D analysis, pose_landmarks for visualization

# CRITICAL: FastAPI async/await patterns
# All video processing must be async to avoid blocking
async def process_video(file: UploadFile):
    # Process in background task to prevent timeout

# CRITICAL: Video format compatibility
# MP4 works best, AVI/MOV may need format conversion
# Use cv2.VideoWriter_fourcc(*'mp4v') for output

# CRITICAL: Pydantic v2 validation patterns
# Follow existing src/perplexity/models.py patterns for consistency
```

## Implementation Blueprint

### Data Models and Structure

Following the established patterns in `src/perplexity/models.py`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

class PoseLandmark(BaseModel):
    """Single 3D pose landmark point."""
    x: float = Field(description="X coordinate in meters")
    y: float = Field(description="Y coordinate in meters") 
    z: float = Field(description="Z coordinate in meters")
    visibility: float = Field(ge=0, le=1, description="Landmark visibility score")

class FrameAnalysis(BaseModel):
    """Analysis of a single video frame."""
    frame_number: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    landmarks: List[PoseLandmark] = Field(max_items=33, min_items=33)
    confidence_score: float = Field(ge=0, le=1)

class VideoAnalysis(BaseModel):
    """Complete video pose analysis results."""
    video_id: str
    total_frames: int = Field(gt=0)
    fps: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    frame_analyses: List[FrameAnalysis]
    processing_time_seconds: float = Field(ge=0)
    gpu_memory_used_mb: float = Field(ge=0)

class CoachingFeedback(BaseModel):
    """AI coach feedback response."""
    analysis_summary: str
    key_issues: List[str]
    improvement_suggestions: List[str]
    confidence_score: float = Field(ge=0, le=1)
    technical_metrics: Dict[str, float] = Field(default_factory=dict)
```

### List of Tasks to Complete the PRP

```yaml
Task 1 - Setup Core Infrastructure:
CREATE src/ai_coach/__init__.py:
  - Basic module initialization
  - Version and metadata

CREATE src/ai_coach/models.py:
  - MIRROR pattern from: src/perplexity/models.py
  - ADD pose-specific Pydantic models above
  - INCLUDE proper validation and field constraints

Task 2 - Implement Pose Analysis Engine:
CREATE src/ai_coach/pose_analyzer.py:
  - MIRROR pattern from: examples/pose.py
  - ENHANCE with 3D landmark extraction
  - ADD GPU memory management
  - INCLUDE batch processing for multiple frames

Task 3 - Build Video Processing Pipeline:
CREATE src/ai_coach/video_processor.py:
  - INTEGRATE FastAPI UploadFile handling
  - ADD temporary file management
  - IMPLEMENT frame extraction and analysis
  - INCLUDE progress tracking and error handling

Task 4 - Develop AI Coach Agent:
CREATE src/ai_coach/coach_agent.py:
  - BUILD conversational AI for pose feedback
  - INTEGRATE with pose analysis results
  - ADD coaching knowledge base
  - IMPLEMENT context-aware responses

Task 5 - Create FastAPI Backend:
CREATE src/ai_coach/api.py:
  - MIRROR pattern from: src/perplexity/cli.py (argument handling)
  - ADD video upload endpoints
  - IMPLEMENT chat interface endpoints
  - INCLUDE WebSocket support for real-time updates

Task 6 - Build Frontend Interface:
CREATE src/frontend/static/index.html:
  - VIDEO upload form with drag-drop
  - CHAT interface for coach interaction
  - PROGRESS indicators for video processing
  - VIDEO player with pose overlay visualization

CREATE src/frontend/static/app.js:
  - HANDLE file uploads and progress
  - IMPLEMENT WebSocket chat connection
  - ADD video playback controls
  - MANAGE pose visualization overlay

CREATE src/frontend/static/style.css:
  - RESPONSIVE design for mobile/desktop
  - MODERN UI following material design principles

Task 7 - Implement Comprehensive Tests:
CREATE tests/test_pose_analyzer.py:
  - MIRROR pattern from: tests/test_evaluator.py
  - TEST pose detection accuracy
  - MOCK MediaPipe for unit tests
  - VALIDATE 3D coordinate calculations

CREATE tests/test_video_processor.py:
  - TEST video format handling
  - MOCK file uploads
  - VALIDATE temporary file management
  - TEST GPU memory cleanup

CREATE tests/test_coach_agent.py:
  - TEST AI response generation
  - MOCK pose analysis input
  - VALIDATE coaching feedback quality

CREATE tests/test_api.py:
  - TEST all FastAPI endpoints
  - MOCK video upload scenarios
  - VALIDATE WebSocket connections
  - TEST error handling

Task 8 - Integration and Performance Optimization:
MODIFY src/ai_coach/pose_analyzer.py:
  - ADD CUDA optimization following RTX 3060 guidelines
  - IMPLEMENT memory-efficient batch processing
  - ADD performance monitoring and metrics

UPDATE pyproject.toml:
  - ADD new dependencies (mediapipe, opencv-python, fastapi, uvicorn)
  - ENSURE compatibility with existing uv setup

Task 9 - Main Application Entry Point:
CREATE src/main.py:
  - MIRROR pattern from: src/perplexity/cli.py (entry point structure)
  - INTEGRATE FastAPI app with pose analysis
  - ADD command-line interface
  - INCLUDE development and production configurations
```

### Per Task Pseudocode

```python
# Task 2 - Pose Analysis Engine
class PoseAnalyzer:
    def __init__(self, use_gpu: bool = True):
        # PATTERN: GPU detection from existing model loading code
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        # CRITICAL: MediaPipe configuration for best accuracy
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.7,    # Higher than examples/pose.py
            min_tracking_confidence=0.5,
            model_complexity=2,              # Highest accuracy
            enable_segmentation=True         # Required for world landmarks
        )
    
    async def analyze_video(self, video_path: str) -> VideoAnalysis:
        # PATTERN: Frame-by-frame processing from examples/pose.py
        cap = cv2.VideoCapture(video_path)
        
        # GOTCHA: Extract video metadata first
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_analyses = []
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # CRITICAL: Color conversion for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # PATTERN: Extract 3D coordinates
                landmarks = [
                    PoseLandmark(
                        x=lm.x, y=lm.y, z=lm.z, 
                        visibility=lm.visibility
                    ) for lm in results.pose_world_landmarks.landmark
                ]
                
                frame_analyses.append(FrameAnalysis(
                    frame_number=frame_number,
                    timestamp_ms=(frame_number / fps) * 1000,
                    landmarks=landmarks,
                    confidence_score=calculate_confidence(landmarks)
                ))
            
            frame_number += 1
            
            # CRITICAL: GPU memory management
            if frame_number % 100 == 0:
                torch.cuda.empty_cache()
        
        cap.release()
        return VideoAnalysis(...)

# Task 3 - Video Processing Pipeline  
class VideoProcessor:
    async def process_upload(self, file: UploadFile) -> str:
        # GOTCHA: FastAPI UploadFile to OpenCV pipeline
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file.filename.split('.')[-1]}"
        )
        
        # PATTERN: Async file handling
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # CRITICAL: Validate video format
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise ValueError("Invalid video format")
        cap.release()
        
        return temp_file.name

# Task 4 - AI Coach Agent
class CoachAgent:
    def __init__(self):
        # PATTERN: Model configuration similar to perplexity system
        self.model = self._load_coaching_model()
        
    async def generate_feedback(self, analysis: VideoAnalysis) -> CoachingFeedback:
        # PATTERN: Analysis processing similar to perplexity evaluator
        metrics = self._calculate_coaching_metrics(analysis)
        
        # CRITICAL: Context-aware coaching prompts
        prompt = self._build_coaching_prompt(analysis, metrics)
        response = await self.model.generate(prompt)
        
        return CoachingFeedback(
            analysis_summary=response.summary,
            key_issues=response.issues,
            improvement_suggestions=response.suggestions,
            confidence_score=response.confidence,
            technical_metrics=metrics
        )
```

### Integration Points
```yaml
DEPENDENCIES (pyproject.toml):
  - add: "mediapipe>=0.10.0"
  - add: "opencv-python>=4.8.0" 
  - add: "fastapi>=0.104.0"
  - add: "uvicorn>=0.24.0"
  - add: "python-multipart>=0.0.6"  # For file uploads
  - add: "websockets>=12.0"         # For real-time chat

ENVIRONMENT:
  - add to: .env template
  - variables: "COACH_MODEL_PATH", "UPLOAD_MAX_SIZE", "GPU_MEMORY_LIMIT"

ROUTES:
  - /upload - POST video upload
  - /analyze/{video_id} - GET analysis results  
  - /chat - WebSocket chat endpoint
  - /videos/{video_id}/preview - GET processed video with overlay

STORAGE:
  - temporary: uploads/ directory for video files
  - results: JSON files or database for analysis results
  - cleanup: background task to remove old files
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
source venv_linux/bin/activate

# Code quality checks
ruff check src/ai_coach/ --fix
mypy src/ai_coach/
ruff format src/ai_coach/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE comprehensive test suite following conftest.py patterns

# tests/test_pose_analyzer.py
def test_pose_detection_accuracy(mock_video_file):
    """Test pose detection returns 33 landmarks."""
    analyzer = PoseAnalyzer(use_gpu=False)  # Use CPU for tests
    result = analyzer.analyze_frame(mock_video_file)
    assert len(result.landmarks) == 33
    assert all(0 <= lm.visibility <= 1 for lm in result.landmarks)

def test_gpu_memory_management(mock_cuda_available):
    """Test GPU memory is properly managed."""
    analyzer = PoseAnalyzer(use_gpu=True)
    initial_memory = torch.cuda.memory_allocated()
    
    # Process multiple frames
    for _ in range(100):
        analyzer.analyze_frame(mock_frame)
    
    final_memory = torch.cuda.memory_allocated()
    assert final_memory < initial_memory + (100 * 1024**2)  # Less than 100MB growth

# tests/test_video_processor.py  
@pytest.mark.asyncio
async def test_video_upload_processing():
    """Test video upload and temporary file creation."""
    mock_file = create_mock_video_file()
    processor = VideoProcessor()
    
    temp_path = await processor.process_upload(mock_file)
    assert os.path.exists(temp_path)
    assert temp_path.endswith('.mp4')
    
    # Cleanup
    os.unlink(temp_path)
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_pose_analyzer.py -v
uv run pytest tests/test_video_processor.py -v 
uv run pytest tests/test_coach_agent.py -v
uv run pytest tests/test_api.py -v

# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Start the FastAPI server
uv run python -m src.main --dev

# Test video upload endpoint
curl -X POST http://localhost:8000/upload \
  -F "video=@examples/test_video.mp4" \
  -H "Content-Type: multipart/form-data"

# Expected: {"video_id": "uuid", "status": "processing"}

# Test analysis results
curl http://localhost:8000/analyze/{video_id}

# Expected: Complete VideoAnalysis JSON with 33 landmarks per frame

# Test chat endpoint via WebSocket
# Use browser console or WebSocket client to test real-time chat
```

### Level 4: Performance Validation
```bash
# Test GPU utilization
nvidia-smi

# Process test video and measure performance
uv run python -c "
from src.ai_coach.pose_analyzer import PoseAnalyzer
import time

analyzer = PoseAnalyzer(use_gpu=True)
start = time.time()
result = analyzer.analyze_video('examples/test_video.mp4')
end = time.time()

print(f'Processing time: {end-start:.2f}s')
print(f'Frames processed: {len(result.frame_analyses)}')
print(f'FPS: {len(result.frame_analyses)/(end-start):.2f}')
"

# Expected: >30 FPS processing speed with GPU
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `ruff check src/ai_coach/`
- [ ] No type errors: `mypy src/ai_coach/`
- [ ] Video upload works via web interface
- [ ] Pose detection extracts 33 landmarks accurately  
- [ ] AI coach provides meaningful feedback
- [ ] Real-time chat responds within 2 seconds
- [ ] GPU processing is >50% faster than CPU
- [ ] Memory usage stays within RTX 3060 limits (12GB)
- [ ] Frontend displays pose overlay visualization
- [ ] Error handling prevents crashes on invalid videos
- [ ] Temporary files are cleaned up properly

## Anti-Patterns to Avoid
- ❌ Don't process entire video in memory - use frame-by-frame streaming
- ❌ Don't block FastAPI endpoints - use background tasks for video processing  
- ❌ Don't ignore MediaPipe coordinate systems - distinguish image vs world coordinates
- ❌ Don't skip GPU memory management - RTX 3060 has limited VRAM
- ❌ Don't hardcode model parameters - use configuration files
- ❌ Don't forget temporary file cleanup - implement proper lifecycle management
- ❌ Don't assume video formats - validate and convert as needed
- ❌ Don't skip pose confidence validation - filter low-quality detections

---

**PRP Confidence Score: 9/10**

This PRP provides comprehensive context, follows established codebase patterns, includes thorough validation loops, and addresses all critical technical gotchas for successful one-pass implementation. The confidence score reflects the depth of research, clear implementation blueprint, and alignment with existing project architecture.