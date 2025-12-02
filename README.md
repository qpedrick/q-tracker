Queue Tracker (Jetson Orin Nano MVP)

Device: NVIDIA Jetson Orin Nano (JetPack 6)
Stack: Python 3.10, YOLOv8 (TensorRT), Supervision, SQLite, Streamlit
Status: Prototype (Headless / NVMe Boot)

Overview

This project is an anonymous, privacy-first queue tracking system. It runs entirely on the edge, counting people within a specific Region of Interest (ROI) and logging the queue length to a local SQLite database. It includes a live web dashboard for monitoring.

Hardware Requirements

Compute: Jetson Orin Nano (8GB recommended)

Storage: NVMe SSD (Boot drive, 500GB+)

Camera: USB Webcam (Logitech C920s or similar)

Network: Ethernet + Tailscale (for headless access)

Software Prerequisites

OS: JetPack 6.x (Ubuntu 22.04)

Environment: Python Virtual Environment (venv) with access to system site packages (for CUDA/TensorRT).

Installation

1. System Dependencies

sudo apt update
sudo apt install -y python3-venv libgl1-mesa-glx libopenblas-base libopenmpi-dev libomp-dev


2. Python Environment Setup

Crucial: Must use --system-site-packages to inherit Jetson's pre-installed TensorRT and OpenCV bindings.

mkdir -p ~/queue_tracker
cd ~/queue_tracker
python3 -m venv venv --system-site-packages
source venv/bin/activate


3. Install Python Libraries

Note: We must pin numpy<2 to ensure compatibility with NVIDIA's PyTorch wheels.

pip install --upgrade pip
pip install "numpy<2"
# Install Jetson-optimized PyTorch
pip install torch torchvision --index-url [https://pypi.jetson-ai-lab.io/jp6/cu126](https://pypi.jetson-ai-lab.io/jp6/cu126) --no-cache-dir
# Install Application Dependencies
pip install ultralytics supervision streamlit plotly pandas py-cpuinfo psutil onnx onnxslim onnxruntime


4. TensorRT Optimization

Convert the standard YOLO model to a TensorRT engine for max FPS on Orin.

yolo export model=yolov8n.pt format=engine device=0 half=True


Running the System

The system consists of two separate processes that must run simultaneously.

1. The Tracker (Background Service)

This script runs the inference loop, logs metrics to SQLite, and saves a debug image.

# Activate venv first!
python queue_tracker.py


Config: Edit QUEUE_REGION in queue_tracker.py to change the polygon zone.

Output: Logs [Timestamp] Queue Length: X to console and queue_metrics.db.

2. The Dashboard (Frontend)

A Streamlit app that visualizes the database and shows the latest debug frame.

streamlit run dashboard.py --server.fileWatcherType none


Access: Open http://<jetson-ip>:8501 in your browser.

Project Structure

queue_tracker.py: Main computer vision pipeline (YOLO + ByteTrack + Zones).

dashboard.py: Visualization UI (reads DB, displays graphs + debug image).

yolov8n.engine: Optimized TensorRT model file.

queue_metrics.db: SQLite database storing time-series counts.

latest_debug.jpg: The most recent frame with bounding boxes drawn (used for calibration).

Troubleshooting

Camera Fail: If /dev/video0 is busy, ensure no other process (like a stuck python script) is using it. sudo fuser -k /dev/video0.

NumPy Errors: If OpenCV or PyTorch crashes, ensure pip show numpy returns a version lower than 2.0 (e.g., 1.26.4).