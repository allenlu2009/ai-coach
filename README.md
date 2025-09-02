# AI Coach - Athlete Pose Analysis System

A comprehensive AI coach system that analyzes athlete poses from uploaded videos and provides conversational feedback through a chat interface. Built using Context Engineering principles for robust, production-ready AI applications.

> **Powered by MediaPipe, FastAPI, and intelligent coaching algorithms.**

## 🚀 Quick Start

```bash
# 1. Clone this repository
git clone <repository-url>
cd ai_coach

# 2. Set up virtual environment with uv (recommended)
uv sync

# 3. Activate the environment
source .venv/bin/activate

# 4. Run the AI coach system
python src/main.py

# 5. Open your browser to http://localhost:8000
# Upload a video and start getting coaching feedback!

# 6. Run tests
pytest tests/ -v
```

## ✨ Features

- **🎥 Video Upload**: Drag-and-drop video upload with support for MP4, AVI, MOV
- **🏃‍♀️ Pose Analysis**: Real-time 33-point pose detection using MediaPipe  
- **🤖 AI Coaching**: Intelligent feedback on form, technique, and movement patterns
- **💬 Chat Interface**: Real-time conversation about your performance
- **⚡ GPU Acceleration**: Optimized for RTX 3060 with memory management
- **📊 Movement Analysis**: Support for squats, deadlifts, overhead press, and more

## 📚 Table of Contents

- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [API Endpoints](#api-endpoints)
- [Development Guide](#development-guide)
- [Testing](#testing)
- [Deployment](#deployment)

## System Architecture

The AI Coach system is built with a modern, scalable architecture:

### 🏗️ High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   AI Engine     │
│   (Web UI)      │◄──►│   Backend       │◄──►│   (Coaching)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Upload  │    │   Pose Analysis │    │   Chat System   │
│   & Display     │    │   (MediaPipe)   │    │   (WebSocket)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🎯 Core Design Principles

1. **Real-time Processing**: Async video analysis with progress tracking
2. **GPU Optimization**: Efficient memory management for RTX 3060
3. **Modular Components**: Separation of concerns for maintainability
4. **Type Safety**: Full Pydantic validation throughout the system

## Project Structure

```
ai_coach/
├── src/
│   ├── ai_coach/
│   │   ├── __init__.py        # Module initialization
│   │   ├── api.py             # FastAPI routes and WebSocket
│   │   ├── coach_agent.py     # AI coaching logic and feedback
│   │   ├── models.py          # Pydantic data models
│   │   ├── pose_analyzer.py   # MediaPipe pose detection
│   │   └── video_processor.py # Video upload and processing
│   ├── frontend/
│   │   └── static/
│   │       ├── index.html     # Web interface
│   │       ├── style.css      # Styling
│   │       └── app.js         # Frontend JavaScript
│   └── main.py               # Application entry point
├── tests/
│   ├── test_coach_agent.py   # AI coaching tests
│   ├── test_pose_analyzer.py # Pose detection tests
│   └── test_video_processor.py # Video processing tests
├── examples/
│   └── pose.py              # MediaPipe reference implementation
├── uploads/                 # Video storage and processing
├── CLAUDE.md               # Development guidelines
└── pyproject.toml          # Project configuration
```

## Core Components

### 🎥 Video Processing Pipeline (`video_processor.py`)
- **Async File Upload**: Handle large video files efficiently
- **Format Validation**: Support MP4, AVI, MOV formats
- **Temporary Storage**: Secure file handling with automatic cleanup
- **Progress Tracking**: Real-time processing status updates

### 🏃‍♀️ Pose Analysis Engine (`pose_analyzer.py`) 
- **MediaPipe Integration**: 33-point pose landmark detection
- **3D Coordinates**: World coordinates for accurate analysis
- **Confidence Scoring**: Quality assessment of pose detection
- **GPU Acceleration**: RTX 3060 optimization with memory management

### 🤖 AI Coach Agent (`coach_agent.py`)
- **Movement Analysis**: Technical metrics (smoothness, stability, balance, symmetry)
- **Intelligent Feedback**: Context-aware coaching recommendations
- **Chat Integration**: Conversational interface for Q&A
- **Movement Patterns**: Specialized analysis for different exercise types

### 🌐 Web API (`api.py`)
- **FastAPI Backend**: High-performance async web server
- **REST Endpoints**: Video upload, analysis results, feedback retrieval
- **WebSocket Chat**: Real-time messaging for coaching conversations
- **Static Files**: Frontend asset serving

## API Endpoints

### REST API
```
POST   /upload          # Upload video for analysis
GET    /analysis/{id}   # Get analysis results
POST   /feedback        # Generate coaching feedback
GET    /chat/{session}  # Get chat history
```

### WebSocket
```
/ws/chat/{session_id}   # Real-time chat interface
```

## Development Guide

### Prerequisites
- Python 3.11+
- RTX 3060 GPU (recommended)
- uv package manager

### Setup Development Environment
```bash
# Install dependencies
uv sync

# Install development tools
uv sync --extra dev

# Run tests
pytest tests/ -v

# Code quality checks
ruff check --fix
mypy src/
```

### Adding New Features
1. **Follow CLAUDE.md guidelines** for coding standards
2. **Reference examples/pose.py** for MediaPipe patterns  
3. **Write tests first** using the existing test patterns
4. **Update models.py** if adding new data structures
5. **Test GPU compatibility** on RTX 3060 hardware

## Testing

### Test Coverage
- **Coach Agent**: 21 tests covering feedback generation and chat
- **Pose Analyzer**: 22 tests covering MediaPipe integration
- **Video Processor**: 21 tests covering upload and processing
- **Integration Tests**: End-to-end workflow testing

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific component
pytest tests/test_coach_agent.py -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# GPU tests only
pytest tests/ -m "gpu" -v
```

## Deployment

### Production Setup
```bash
# Production dependencies
uv sync --no-dev

# Run with production server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# With GPU optimization
CUDA_VISIBLE_DEVICES=0 uvicorn src.main:app
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install uv && uv sync --no-dev
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
HOST=0.0.0.0
PORT=8000
DEBUG=false
GPU_ENABLED=true
MAX_VIDEO_SIZE_MB=100
```

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 8GB
- Storage: 20GB free space

### Recommended 
- GPU: RTX 3060 (12GB VRAM)
- CPU: 8+ cores
- RAM: 16GB
- Storage: 50GB+ for video processing

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow CLAUDE.md coding guidelines
4. Write tests for new functionality
5. Submit pull request

---

Built with ❤️ using Context Engineering principles for robust AI applications.