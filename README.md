# Intelligent Traffic Junction Control System

Real-time traffic monitoring and adaptive signal control using YOLOv11, ByteTrack, and intelligent reasoning.

## Features

- 🚗 Real-time vehicle detection with YOLOv11
- 📍 Multi-object tracking with ByteTrack
- 🧠 4-priority decision hierarchy
- ⏱️ Starvation prevention (no lane waits >15s)
- 📊 Pressure balancing for congestion
- 🎥 Live visualization

## Installation
```bash
pip install opencv-python numpy ultralytics
```

## Usage
```bash
python detect.py
```

Update `camera_id` in the code to use your video file:
```python
camera_id='your_video.mp4'  # or 0 for webcam
```

## How It Works

1. **YOLOv11** detects vehicles in each frame
2. **ByteTrack** maintains vehicle IDs across frames
3. **Reasoning Engine** analyzes traffic and decides signal timing
4. System adapts signals based on:
   - Starvation Prevention (Priority 2)
   - Pressure Balancing (Priority 3)
   - Normal Rotation (Priority 4)

## Current Status

- ✅ Vehicle detection and tracking
- ✅ Adaptive signal timing
- ✅ Real-time visualization
- ⚠️ Emergency vehicle detection (TODO)

## Output Example
```
PRIORITY 2 ACTIVATED: STARVATION PREVENTION
  Lane W has waited 59.8s
  Action: Switch to EW phase
  Duration: 30.0s
```

## Future Improvements

- Emergency vehicle preemption
- Pedestrian/cyclist detection
- Stop line enforcement
- Historical pattern learning