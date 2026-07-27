"""Configuration for the live ear-landmark pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Models
MODEL = "shgnet"  # "shgnet" | "cnn"
YOLO_WEIGHTS = str(ROOT / "models" / "yolo" / "yolo26n-pose.pt")
YOLO_ONNX = str(ROOT / "models" / "yolo" / "yolo26n-pose.onnx")
SHGNET_CHECKPOINT = str(ROOT / "models" / "shgnet" / "hourglass_2stack_best.pth")
SHGNET_ONNX = str(ROOT / "models" / "shgnet" / "hourglass_2stack.onnx")
CNN_WEIGHTS = str(ROOT / "models" / "cnn" / "my_model.h5")
MEDIAPIPE_MODEL = str(ROOT / "models" / "face_landmarker.task")
FACE_DETECTOR_ONNX = str(ROOT / "models" / "mediapipe_onnx" / "face_detector.onnx")
FACE_LANDMARKS_ONNX = str(
    ROOT / "models" / "mediapipe_onnx" / "face_landmarks_detector.onnx"
)
INFERENCE_BACKEND = "onnx"  # "torch" | "onnx" — live app uses ONNX

# Detection cadence
YOLO_EVERY_N = 5
YOLO_CONF = 0.35
YOLO_IOU = 0.45
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5
MP_MIN_PRESENCE_CONFIDENCE = 0.5

# Ear ROI (YOLO tip; no MediaPipe)
# ROI height = face_h × ROI_FACE_HEIGHT_RATIO × ROI_PAD
# ROI width  = height × ROI_ASPECT
ROI_FACE_HEIGHT_RATIO = 0.55
ROI_ASPECT = 0.78
ROI_PAD = 1.15
EAR_KEYPOINT_MIN_CONF = 0.25
YAW_EAR_THRESHOLD_DEG = 12.0  # unused (no MediaPipe yaw)

# Temporal smoothing (One Euro Filter for landmarks; EMA for ROI box)
# Balanced: low rest jitter via freeze + beta>0 so fast head turns catch up.
# Rest freeze holds when filtered median speed < REST_SPEED_PX (px/s).
EMA_ROI = 0.35
ONE_EURO_MIN_CUTOFF = 0.5
ONE_EURO_BETA = 0.007
ONE_EURO_D_CUTOFF = 1.0
ONE_EURO_REST_SPEED_PX = 8.0
ONE_EURO_REST_HOLD_FRAMES = 2
LOST_TRACK_FRAMES = 20

# Landmark model I/O
SHGNET_INPUT_SIZE = 256
SHGNET_HEATMAP_SIZE = 64
NUM_LANDMARKS = 55
CNN_INPUT_SIZE = 224

# Display
WINDOW_NAME = "Ear Landmark Live"
DRAW_LANDMARK_INDICES = False
LANDMARK_RADIUS = 2
ROI_COLOR = (80, 200, 120)
LANDMARK_COLOR = (0, 220, 255)
HUD_COLOR = (240, 240, 240)
