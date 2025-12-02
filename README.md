# QTracker

**Privacy-first, edge-based queue monitoring system for NVIDIA Jetson Orin Nano**

![Status](https://img.shields.io/badge/status-MVP-orange)
![Platform](https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-76B900)
![Python](https://img.shields.io/badge/python-3.10-blue)

## Overview

An anonymous queue tracking system that runs entirely on-device. Uses computer vision to count people within a defined Region of Interest (ROI), storing metrics locally with zero cloud dependencies. Perfect for retail, events, or public spaces where privacy matters.

**Key Features:**
- 🔒 **Privacy-First**: No cloud uploads, all processing on-device
- ⚡ **Real-Time**: TensorRT-optimized YOLOv8 inference
- 📊 **Live Dashboard**: Web-based monitoring via Streamlit
- 💾 **Local Storage**: SQLite time-series database
- 🎯 **Configurable ROI**: Define custom queue zones

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **Compute** | NVIDIA Jetson Orin Nano (8GB recommended) |
| **Storage** | NVMe SSD (500GB+, boot drive) |
| **Camera** | USB Webcam (Logitech C920s or similar) |
| **Network** | Ethernet + Tailscale (for remote access) |

## Software Stack

- **OS**: JetPack 6.x (Ubuntu 22.04)
- **Runtime**: Python 3.10 with system site-packages
- **Detection**: YOLOv8n (TensorRT optimized)
- **Tracking**: ByteTrack via Supervision
- **Database**: SQLite
- **Frontend**: Streamlit + Plotly

## Installation

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y python3-venv libgl1-mesa-glx libopenblas-base libopenmpi-dev libomp-dev
```

### 2. Python Environment

⚠️ **Critical**: Use `--system-site-packages` to access Jetson's CUDA/TensorRT bindings.

```bash
cd ~/queue-tracker
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install "numpy<2"  # Required for NVIDIA wheel compatibility

# Jetson-optimized PyTorch
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir

# Application dependencies
pip install ultralytics supervision streamlit plotly pandas py-cpuinfo psutil onnx onnxslim onnxruntime
```

### 4. Optimize Model for TensorRT

```bash
yolo export model=yolov8n.pt format=engine device=0 half=True
```

This creates `yolov8n.engine` - a TensorRT-optimized model for maximum FPS.

## Usage

The system requires **two concurrent processes**:

### 1. Start the Tracker (Backend)

```bash
source venv/bin/activate
python queue_tracker.py
```

**Configuration**: Edit `QUEUE_REGION` in `queue_tracker.py` to define your ROI polygon.

**Output**: 
- Console logs: `[Timestamp] Queue Length: X`
- Database: `queue_metrics.db`
- Debug frame: `latest_debug.jpg`

### 2. Start the Dashboard (Frontend)

```bash
streamlit run dashboard.py --server.fileWatcherType none
```

**Access**: Navigate to `http://<jetson-ip>:8501` in your browser.

## Project Structure

```
queue-tracker/
├── queue_tracker.py      # Main CV pipeline (YOLO + ByteTrack)
├── dashboard.py          # Streamlit visualization UI
├── yolov8n.engine        # TensorRT model (generated)
├── queue_metrics.db      # SQLite database (runtime)
├── latest_debug.jpg      # Latest annotated frame (runtime)
└── README.md
```

## Configuration

### Customize Queue Region

Edit the polygon coordinates in `queue_tracker.py`:

```python
QUEUE_REGION = np.array([
    [400, 300],   # Top-left
    [800, 300],   # Top-right
    [800, 600],   # Bottom-right
    [400, 600]    # Bottom-left
])
```

Use `latest_debug.jpg` to visualize and adjust your region of interest (ROI).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Camera not found** | Check `/dev/video0` exists. Kill stuck processes: `sudo fuser -k /dev/video0` |
| **NumPy version errors** | Ensure `numpy<2.0` installed: `pip show numpy` should show `1.26.x` |
| **Low FPS** | Verify TensorRT engine was created (not using `.pt` model) |
| **Dashboard not updating** | Ensure `queue_tracker.py` is running and writing to database |

## Performance

- **Inference Speed**: ~30-45 FPS on Orin Nano (TensorRT FP16)
- **Memory Usage**: ~2.5GB total (model + tracking)
- **Power**: ~10W average

## Roadmap

- [ ] Heatmap visualization
- [ ] Multi-zone support
- [ ] Alert system (queue threshold notifications)
- [ ] Historical analytics
- [ ] Docker containerization

## License

MIT

## Author

Quinton Ulysses Pedrick

## AI Credits

- [x] Gemini 3
- [x] ChatGPT 5.1
- [x] Copilot