#!/usr/bin/env python3
"""
Simple server runner that avoids MediaPipe initialization issues.
"""

import os
import sys
import argparse
from pathlib import Path
import threading
import time

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Try GPU first, fallback to CPU if needed
# Comment out the GPU disable to allow RTX 3060 acceleration
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # Use CPU only to avoid GPU issues
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # Disable hardware transforms for stability

def _inspect_video_file(path: Path):
	"""
	Attempt to open the video with OpenCV and report:
	- file size
	- frame count, fps, width/height (if available)
	- whether the first frame contains non-black pixels (simple proxy that something is drawn)
	"""
	try:
		import cv2
	except Exception:
		print(f"🟨 Inspect: OpenCV not available; cannot inspect {path.name}")
		return

	if not path.exists():
		print(f"🟥 Inspect: {path} does not exist")
		return

	try:
		file_size = path.stat().st_size
		cap = cv2.VideoCapture(str(path))
		if not cap.isOpened():
			print(f"🟥 Inspect: Failed to open video {path.name} with OpenCV")
			return

		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

		ret, frame = cap.read()
		mean_val = None
		if ret and frame is not None:
			# mean intensity across channels; if near zero likely blank/black
			mean_val = float(frame.mean())

		cap.release()

		print(f"🟦 Inspect: {path.name} size={file_size}B frames={frame_count} fps={fps:.2f} "
			  f"resolution={width}x{height} first_frame_mean={mean_val}")
		if mean_val is not None and mean_val < 1.0:
			print("⚠️ Inspect: first frame is nearly black — pose overlay may not have been drawn or encoding produced black frames.")
		elif mean_val is not None:
			print("✅ Inspect: first frame has non-zero pixels (visual content detected).")
	except Exception as e:
		print(f"❌ Inspect: Error while inspecting {path.name}: {e}")


def _watch_uploads_dir(uploads_dir: Path, poll_interval: float = 2.0):
	"""
	Background watcher that logs newly created MP4 files and inspects them.
	Provides quick diagnostics to help determine whether final-video-with-pose files are valid.
	"""
	print(f"🔎 Uploads watcher starting for: {uploads_dir}")
	seen = set()
	while True:
		try:
			for p in uploads_dir.glob("*.mp4"):
				if p not in seen:
					seen.add(p)
					print(f"📥 New upload detected: {p.name}")
					_inspect_video_file(p)
		except Exception as e:
			print(f"❌ Uploads watcher error: {e}")
		time.sleep(poll_interval)


def main():
	"""Run the AI Coach server with stability fixes."""
	try:
		# Parse command line arguments
		parser = argparse.ArgumentParser(description='AI Coach - Pose Analysis System')
		parser.add_argument('--gpu-encoding', action='store_true', 
						   help='Enable GPU-accelerated FFmpeg encoding (requires NVIDIA GPU with NVENC)')
		parser.add_argument('--create-video', action='store_true',
						   help='Create video overlay files (default: JSON-only for 3x faster results)')
		parser.add_argument('--frame-skip', type=int, default=3,
						   help='Analyze every Nth frame for speedup (default: 3 for 3x faster analysis)')
		parser.add_argument('--use-3d', action='store_true',
						   help='Enable 3D pose estimation (default: 2D pose detection)')
		parser.add_argument('--host', default='127.0.0.1', 
						   help='Host to bind the server to (default: 127.0.0.1)')
		parser.add_argument('--port', type=int, default=8000,
						   help='Port to bind the server to (default: 8000)')
		
		args = parser.parse_args()
		
		import uvicorn
		from dotenv import load_dotenv

		# Load environment variables
		load_dotenv()
		
		# Set 3D pose environment variable based on command line argument
		if args.use_3d:
			os.environ['USE_3D_POSE'] = 'true'
		else:
			os.environ['USE_3D_POSE'] = 'false'

		# Ensure uploads directory exists (use project-local uploads folder)
		uploads_dir = Path(__file__).parent / "uploads"
		uploads_dir.mkdir(exist_ok=True)

		# Log effective environment hints that often affect MediaPipe/openCV behavior
		print(f"📂 Uploads directory: {uploads_dir.resolve()}")
		print(f"🌐 MEDIAPIPE_DISABLE_GPU={os.getenv('MEDIAPIPE_DISABLE_GPU')}")
		print(f"🔧 OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS={os.getenv('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS')}")
		print(f"🔮 USE_3D_POSE={os.getenv('USE_3D_POSE')}")

		# Start background watcher to help diagnose final video files produced at runtime
		watcher = threading.Thread(target=_watch_uploads_dir, args=(uploads_dir,), daemon=True)
		watcher.start()

		print("🏃‍♀️ Starting AI Coach Server...")
		print(f"📍 Server will be available at: http://{args.host}:{args.port}")
		print("🚀 GPU acceleration enabled for RTX 3060")
		if args.gpu_encoding:
			print("🎬 GPU-accelerated FFmpeg encoding enabled (NVIDIA NVENC)")
		else:
			print("🖥️ CPU-based FFmpeg encoding (use --gpu-encoding for GPU acceleration)")
		
		if args.create_video:
			print("🎥 Video overlay creation enabled (slower but creates MP4 files)")
		else:
			print("⚡ JSON-only mode enabled (3x faster, no video overlay files)")
		
		print(f"🎯 Frame skipping: analyzing every {args.frame_skip} frames (effective speedup: ~{args.frame_skip}x)")
		
		if args.use_3d:
			print("🔮 3D pose estimation enabled - generating X,Y,Z coordinates")
		else:
			print("📊 2D pose detection mode (default) - use --use-3d for 3D coordinates")

		# Import the app creation function and pass all optimization parameters
		from ai_coach.api import create_app
		app = create_app(
			uploads_dir=str(uploads_dir), 
			use_gpu_encoding=args.gpu_encoding,
			create_video_overlay=args.create_video,
			frame_skip=args.frame_skip
		)

		# Run with the configured app instance
		uvicorn.run(
			app,
			host=args.host,
			port=args.port,
			reload=False,  # Disable reload to avoid MediaPipe re-init issues
			log_level="info"
		)

	except Exception as e:
		print(f"❌ Server startup failed: {e}")
		print("💡 Try running: pip install uvicorn[standard]")
		sys.exit(1)

if __name__ == "__main__":
	main()