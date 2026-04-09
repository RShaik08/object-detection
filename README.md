# Intelligent Traffic Junction Control System

Real-time adaptive traffic signal control using YOLOv11, ByteTrack, and priority-based reasoning.

## 🎯 Project Overview

This system transforms traditional fixed-time traffic signals into an intelligent adaptive system that:
- Detects vehicles in real-time using YOLOv11
- Tracks vehicles across frames with ByteTrack
- Makes intelligent signal timing decisions using a 4-priority hierarchy
- Prevents lane starvation (no lane waits >15s)
- Balances traffic pressure dynamically
- Provides emergency vehicle preemption

## 📊 System Architecture
```
Video Input → YOLOv11 Detection → ByteTrack → Priority Reasoning → Signal Control
                                      ↓
                              Live Visualization
```

## 🚀 Quick Start

### Installation
```bash
pip install opencv-python numpy ultralytics
```

### Run
```bash
python detect.py
```

### Controls
- `q` - Quit
- `r` - Show reasoning
- `s` - Show statistics
- `e` - Simulate emergency vehicle (for demo)

## 📈 Performance Metrics

Tested on 3-minute traffic footage:
- **Total Detections:** 5,757
- **Phase Changes:** 32
- **Processing Speed:** 30 FPS (real-time)
- **Starvation Events Detected:** 100%
- **Average Response Time:** <3 seconds

## 🎓 Research Contributions

1. **Three-tier priority framework** with quantifiable thresholds
2. **Real-time performance** suitable for production deployment
3. **Interpretable decisions** with chain-of-thought reasoning
4. **Validated on real traffic** footage with measurable outcomes


## 🔮 Future Work

- Custom YOLO training for ambulance detection
- Multi-modal safety (pedestrians, cyclists)
- Network-level coordination
- Historical pattern learning

## 📧 Contact

Rida Shaik - ridashaik.08@gmail.com
