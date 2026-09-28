# 💎 Real-Time AR Ear Jewellery Virtual Try-On

An end-to-end Computer Vision and Augmented Reality (AR) system for real-time ear jewellery virtual try-on from live webcam or video feeds. The pipeline integrates deep-learning pose estimation, fine-grained 55-point anatomical ear landmark regression, temporal jitter reduction, and live overlay compositing across both **Desktop (Python/OpenCV)** and **Web (JavaScript/ONNX Runtime Web)** runtimes.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=flat-square)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Inference-005CED?style=flat-square&logo=onnx)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-Web_App-339933?style=flat-square&logo=node.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🌟 Highlights

- ⚡ **Real-Time Performance**: Runs at **~25–30 FPS** on consumer hardware with low latency.
- 🎯 **Two-Stage Vision Pipeline**:
  1. **YOLO Pose Estimation**: Detects ear tip keypoints and head orientation to define stable, scale-adaptive bounding crops.
  2. **Stacked Hourglass Network (SHGNet)**: High-resolution heatmap regression extracting **55 anatomical ear landmarks** covering the outer helix, antihelix, tragus, antitragus, concha, and earlobe.
- 🌊 **Jitter-Free Temporal Filtering**: Implements a customized **One Euro Filter** with adaptive speed-based freeze thresholds and exponential moving average (EMA) ROI bounding box tracking.
- 🖥️ **Dual Deployment Targets**:
  - **Desktop Application**: High-performance Python backend leveraging ONNX Runtime CPU/CUDA and OpenCV GUI.
  - **Web Application**: Zero-install browser experience powered by Node.js, WebSockets, and ONNX Runtime in JavaScript.
- 📦 **Exported ONNX Models**: Pre-optimized ONNX models included for zero-friction inference without heavyweight training dependencies.

---

## 🏗️ Architecture & Pipeline Flow

```
Live Video / Webcam Feed (1280x720)
       │
       ▼
[Stage 1: YOLO Pose Keypoint Detector]  ── (Every N frames cadence)
       │
       ▼  (Ear tip coordinates & head orientation)
[Scale-Adaptive Square ROI Extraction]  ── (Pinna height framing)
       │
       ▼  (Cropped & normalized ROI tensor: 256x256)
[Stage 2: 2-Stack Hourglass Network (SHGNet)]  ── (ONNX Runtime engine)
       │
       ▼  (55 Ear Landmark Heatmaps: 64x64)
[Sub-Pixel Argmax & Affine Inverse Mapping]
       │
       ▼  (Raw 2D landmark coordinates)
[One Euro Filter & Temporal Smoothing]  ── (Rest-speed freeze & low-pass filtering)
       │
       ▼  (Stabilized anatomical landmarks)
[AR Jewellery Compositing & HUD Renderer]  ── (Earring positioning, scaling & alpha blending)
```

### 55-Point Ear Landmark Topology
The model regresses 55 anatomical landmarks representing the complete geometry of the human ear:
- **Outer Helix (1–14)**: Tracing the curvature of the outer rim from root to lobe transition.
- **Antihelix & Crura (15–26)**: Superior and inferior crus defining inner cartilage structure.
- **Tragus & Antitragus (27–36)**: Anterior ear canal boundaries for ear-cuff and stud anchoring.
- **Concha Bowl (37–44)**: Deep cavity depth markers.
- **Lobule / Earlobe (45–55)**: Precision contact points for dangling earrings, studs, and hoops.

---

## 📂 Project Structure

```
ear-jewellery-virtual-try-on/
├── app.py                      # Desktop application entry point
├── config.py                   # Centralized pipeline configuration & thresholds
├── requirements.txt            # Python dependencies
├── detectors/                  # Detection & inference wrappers
│   ├── yolo_pose.py            # YOLO ear keypoint detection
│   └── shgnet_onnx.py          # 2-Stack Hourglass ONNX landmark inference
├── models/                     # Pre-trained optimized models
│   ├── yolo/                   # YOLO pose detector (.onnx & .pt)
│   └── shgnet/                 # 2-Stack Hourglass landmark model (.onnx)
├── tracking/                   # Temporal smoothing algorithms
│   └── one_euro.py             # One Euro filter implementation
├── utils/                      # Helper modules
│   ├── coordinates.py          # Bounding box & affine transformation math
│   └── visualization.py        # Heads-up display (HUD) & overlay rendering
└── web/                        # Web / Browser application
    ├── server.mjs              # Node.js web server
    ├── index.html              # Frontend user interface
    ├── infer.js                # Browser inference orchestrator
    ├── ear_roi.js              # JavaScript ROI calculation
    ├── one_euro.js             # JavaScript One Euro filter
    └── package.json            # Node.js dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Webcam / video capture device
- Node.js 18+ (for Web app only)

---

### Option A: Desktop Application (Python)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/santoshnarreddy/ear-jewellery-virtual-try-on.git
   cd ear-jewellery-virtual-try-on
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the live desktop application:**
   ```bash
   python app.py
   ```

**Runtime Controls:**
| Key | Action |
|:---:|:---|
| `q` | Exit application |
| `d` | Toggle debug visualizer (ROI boxes, raw heatmaps, landmark IDs) |
| `c` | Switch active camera input index |

---

### Option B: Web Application (Browser / Node.js)

1. **Navigate to the web folder:**
   ```bash
   cd web
   ```

2. **Install npm packages:**
   ```bash
   npm install
   ```

3. **Start the local server:**
   ```bash
   npm start
   ```

4. **Open in your browser:**
   Navigate to [http://127.0.0.1:8765](http://127.0.0.1:8765) and allow camera permissions.

---

## ⚙️ Key Configuration Options

All pipeline settings can be tuned in `config.py`:

```python
# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection cadence
YOLO_EVERY_N = 5             # Run YOLO keypoint detector every N frames
YOLO_CONF = 0.35             # Confidence threshold for ear keypoint

# Temporal smoothing (One Euro Filter)
EMA_ROI = 0.35               # Exponential smoothing factor for bounding box
ONE_EURO_MIN_CUTOFF = 0.5    # Minimum cutoff frequency (reduces slow-speed jitter)
ONE_EURO_BETA = 0.007        # Speed coefficient (tracks fast movements without lag)
ONE_EURO_REST_SPEED_PX = 8.0 # Pixel speed threshold to lock landmarks during rest
```

---

## 🔬 Benchmark & Latency

Tested on Apple Silicon / Intel Core i7 (CPU only, ONNX Runtime backend):

| Pipeline Stage | Model / Algorithm | Avg Latency |
|---|---|:---:|
| Pose Detection | YOLO pose (onnx) | ~14 ms |
| Landmark Regression | 2-Stack Hourglass (onnx) | ~18 ms |
| Temporal Filter | One Euro + EMA | < 0.5 ms |
| AR Compositing | Alpha blending | ~1.5 ms |
| **Total Pipeline** | **End-to-End** | **~34 ms (~30 FPS)** |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Santosh Narreddy**  
AI/ML Engineer · Computer Vision · Deep Learning  
- GitHub: [@santoshnarreddy](https://github.com/santoshnarreddy)  
- LinkedIn: [santoshnarreddy](https://linkedin.com/in/santoshnarreddy)  
