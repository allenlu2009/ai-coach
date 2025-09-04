# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🏃‍♀️ AI Coach - Video Pose Analysis System

This project is an AI coach system that analyzes athlete poses from uploaded videos and provides conversational feedback through a chat interface.

## 🚀 Development Commands

### Virtual Environment & Package Management
```bash
# Activate virtual environment (Linux/WSL)
source venv_linux/bin/activate

# Install dependencies with uv (preferred)
uv sync

# Alternative: pip install
pip install -e .
```

### Testing & Quality Assurance
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_coach_agent.py -v

# Type checking with mypy
mypy src/

# Code formatting and linting
ruff check --fix
ruff format
```

## 🏗️ Project Architecture

### Core Components

**Video Pose Analysis**: Computer vision pipeline using MediaPipe and OpenCV:
- Real-time pose detection and landmark extraction
- 3D coordinate analysis for athletic movement assessment
- Video processing with pose overlay visualization
- Support for various video formats (MP4, etc.)

**Frontend**: Web interface for video upload and chat interaction
- Video upload functionality for athletic performance analysis
- Real-time chat interface for coaching feedback
- Video playback with pose visualization overlay

**Backend**: AI-powered coaching analysis system
- Conversational AI for pose analysis and feedback
- Integration with pose detection pipeline
- Chat-based interaction for personalized coaching

### Key Files

**examples/pose.py**: Reference implementation showing:
- `display_video_with_pose()` - Real-time pose detection
- `display_save_video_with_pose()` - Process and save analyzed video
- `display_save_video_with_pose_3d()` - 3D landmark extraction
- MediaPipe integration patterns for pose analysis

### Hardware Optimization
- **RTX 3060 GPU support** - Leverage CUDA acceleration for video processing
- **12GB VRAM optimization** - Efficient memory management for real-time analysis
- **GPU-accelerated pose detection** - Use available GPU resources when needed

## 🔄 Project Awareness & Context
- **Check `INITIAL.md`** to understand current feature requirements focused on AI coaching
- **Use venv_linux** (the virtual environment) for all Python commands and tests
- **Use uv for package management** when available, fall back to pip
- **Reference examples/pose.py** for MediaPipe and OpenCV patterns
- **Leverage RTX 3060 GPU** for video processing and AI model inference when needed

## 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For AI coaching components this looks like:
    - `video_analyzer.py` - Video processing and pose detection logic
    - `coach_agent.py` - AI coaching conversation and analysis
    - `pose_utils.py` - Pose landmark processing utilities
    - `frontend.py` - Web interface for video upload and chat
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_dotenv()** for environment variables.

## 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

## ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a "Discovered During Work" section.

## 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write **docstrings for every function** using the Google style:
  ```python
  def analyze_pose():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

## 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

## 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.

## 🎯 AI Coach Specific Guidelines
- **Focus on athlete pose analysis** - All features should support video-based coaching
- **Use MediaPipe patterns from examples/pose.py** - Follow established pose detection approaches
- **Implement conversational coaching** - Chat interface should provide meaningful athletic feedback
- **Leverage GPU acceleration** - Use CUDA when available for faster video processing
- **Support real-time analysis** - Design for responsive video processing and feedback