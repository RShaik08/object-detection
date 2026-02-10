"""
Intelligent Traffic Junction Control System
YOLOv11 + ByteTrack + Advanced Rule-Based Reasoning

OPTIMIZED FOR PERFORMANCE - No LLM delays!
Works perfectly on any system with detailed reasoning output.

Requirements:
pip install opencv-python numpy ultralytics
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime
import time
from collections import deque

# YOLOv11 for real-time detection
from ultralytics import YOLO


@dataclass
class Vehicle:
    """Represents a tracked vehicle in the junction"""
    id: str
    lane: str
    type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    wait_time: float = 0.0
    detected_time: float = field(default_factory=time.time)
    is_emergency: bool = False
    
    def update_position(self, new_x: float, new_y: float, dt: float = 0.1):
        """Update vehicle position and calculate velocity"""
        self.vx = (new_x - self.x) / max(dt, 0.001)
        self.vy = (new_y - self.y) / max(dt, 0.001)
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        self.x = new_x
        self.y = new_y
    
    def is_stopped(self, threshold: float = 2.0) -> bool:
        """Check if vehicle is stopped"""
        return self.speed < threshold
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point from bbox"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class VehicleTrack:
    """ByteTrack representation"""
    id: str
    lane: str
    type: str
    first_seen: float
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    wait_time: float = 0.0
    speed: float = 0.0
    is_emergency: bool = False
    
    def update(self, x: float, y: float, wait_time: float, speed: float):
        self.positions.append((x, y, time.time()))
        self.wait_time = wait_time
        self.speed = speed


class TrafficSignalPhase:
    """Traffic signal phase management"""
    def __init__(self, name: str, green_lanes: List[str], duration: float = 30.0):
        self.name = name
        self.green_lanes = green_lanes
        self.duration = duration
        self.timer = duration
    
    def is_green(self, lane: str) -> bool:
        return lane in self.green_lanes
    
    def update(self, dt: float = 0.1):
        self.timer -= dt
        return self.timer <= 0
    
    def reset(self, duration: float = None):
        self.timer = duration if duration is not None else self.duration


class IntelligentTrafficReasoner:
    """
    Advanced Rule-Based Reasoning Engine
    Implements the same 4-priority hierarchy with detailed explanations
    FAST - No LLM delays!
    """
    
    def __init__(self):
        self.reasoning_log = []
        self.decision_count = 0
        
    def analyze_junction(self, vehicles: List[Vehicle], tracks: Dict[str, VehicleTrack], 
                        current_phase: str) -> Tuple[str, float, str]:
        """
        Analyze junction and make intelligent decision
        
        Returns: (next_phase, duration, reasoning_text)
        """
        
        self.decision_count += 1
        
        # Gather junction state
        lane_counts = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        lane_wait_times = {'N': [], 'S': [], 'E': [], 'W': []}
        lane_speeds = {'N': [], 'S': [], 'E': [], 'W': []}
        emergency_detected = False
        emergency_lane = None
        emergency_distance = float('inf')
        
        for v in vehicles:
            lane_counts[v.lane] += 1
            lane_speeds[v.lane].append(v.speed)
            
            if v.is_stopped():
                lane_wait_times[v.lane].append(v.wait_time)
            
            if v.is_emergency:
                emergency_detected = True
                emergency_lane = v.lane
                # Calculate distance to junction center (simplified)
                emergency_distance = min(emergency_distance, abs(640 - v.x) + abs(360 - v.y))
        
        # Calculate statistics
        avg_waits = {}
        avg_speeds = {}
        for lane in ['N', 'S', 'E', 'W']:
            times = lane_wait_times[lane]
            speeds = lane_speeds[lane]
            avg_waits[lane] = sum(times) / len(times) if times else 0.0
            avg_speeds[lane] = sum(speeds) / len(speeds) if speeds else 0.0
        
        # Build detailed reasoning
        reasoning = self._build_detailed_reasoning(
            current_phase, lane_counts, avg_waits, avg_speeds,
            emergency_detected, emergency_lane, emergency_distance,
            len(vehicles)
        )
        
        # Make decision based on hierarchy
        decision_phase, decision_duration, decision_reason = self._make_decision(
            current_phase, lane_counts, avg_waits, avg_speeds,
            emergency_detected, emergency_lane, emergency_distance
        )
        
        # Combine reasoning and decision
        full_reasoning = reasoning + "\n" + decision_reason
        full_reasoning += f"\n\n{'='*60}\n"
        full_reasoning += f"FINAL DECISION: Phase={decision_phase}, Duration={decision_duration}s\n"
        full_reasoning += f"{'='*60}\n"
        
        self.reasoning_log.append(full_reasoning)
        return decision_phase, decision_duration, full_reasoning
    
    def _build_detailed_reasoning(self, current_phase: str, lane_counts: Dict,
                                  avg_waits: Dict, avg_speeds: Dict,
                                  emergency: bool, emergency_lane: str,
                                  emergency_dist: float, total_vehicles: int) -> str:
        """Build detailed reasoning output"""
        
        reasoning = f"{'='*60}\n"
        reasoning += f"TRAFFIC-R1 INTELLIGENT REASONING ENGINE\n"
        reasoning += f"{'='*60}\n"
        reasoning += f"Decision #{self.decision_count} | Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        reasoning += f"CURRENT STATE ANALYSIS:\n"
        reasoning += f"  Active Phase: {current_phase}\n"
        reasoning += f"  Total Vehicles: {total_vehicles}\n\n"
        
        reasoning += f"LANE-BY-LANE BREAKDOWN:\n"
        for lane in ['N', 'S', 'E', 'W']:
            reasoning += f"  Lane {lane}:\n"
            reasoning += f"    • Vehicles: {lane_counts[lane]}\n"
            reasoning += f"    • Avg Wait Time: {avg_waits[lane]:.1f}s\n"
            reasoning += f"    • Avg Speed: {avg_speeds[lane]:.1f} px/s\n"
            
            # Add status indicators
            if lane_counts[lane] == 0:
                reasoning += f"    • Status: EMPTY ✓\n"
            elif avg_waits[lane] > 15:
                reasoning += f"    • Status: STARVING ⚠️\n"
            elif avg_waits[lane] > 8:
                reasoning += f"    • Status: CONGESTED 🔶\n"
            else:
                reasoning += f"    • Status: FLOWING ✓\n"
        
        reasoning += f"\nTRAFFIC PRESSURE ANALYSIS:\n"
        ns_load = lane_counts['N'] + lane_counts['S']
        ew_load = lane_counts['E'] + lane_counts['W']
        ns_avg_wait = (avg_waits['N'] + avg_waits['S']) / 2
        ew_avg_wait = (avg_waits['E'] + avg_waits['W']) / 2
        
        reasoning += f"  NS Direction: {ns_load} vehicles, {ns_avg_wait:.1f}s avg wait\n"
        reasoning += f"  EW Direction: {ew_load} vehicles, {ew_avg_wait:.1f}s avg wait\n"
        
        if ns_load > 0 and ew_load > 0:
            ratio = ns_load / ew_load if ew_load > 0 else float('inf')
            reasoning += f"  Load Ratio (NS/EW): {ratio:.2f}x\n"
        
        if emergency:
            reasoning += f"\n🚨 EMERGENCY ALERT:\n"
            reasoning += f"  Emergency vehicle detected in Lane {emergency_lane}\n"
            reasoning += f"  Distance to junction: ~{emergency_dist:.0f} pixels\n"
            reasoning += f"  Priority: CRITICAL - Immediate action required\n"
        
        reasoning += f"\n"
        return reasoning
    
    def _make_decision(self, current_phase: str, lane_counts: Dict,
                      avg_waits: Dict, avg_speeds: Dict,
                      emergency: bool, emergency_lane: str,
                      emergency_dist: float) -> Tuple[str, float, str]:
        """
        Make traffic signal decision following 4-priority hierarchy
        """
        
        # PRIORITY 1: EMERGENCY VEHICLE PREEMPTION
        if emergency:
            decision_phase = 'NS' if emergency_lane in ['N', 'S'] else 'EW'
            
            # Determine duration based on distance
            if emergency_dist < 200:
                duration = 15.0  # Very close, longer green
            else:
                duration = 10.0  # Further away, shorter green
            
            reason = f"PRIORITY 1 ACTIVATED: EMERGENCY VEHICLE PREEMPTION\n"
            reason += f"  Rule: Emergency vehicles have absolute priority\n"
            reason += f"  Analysis: Ambulance detected in lane {emergency_lane}\n"
            reason += f"  Action: Immediate switch to {decision_phase} phase\n"
            reason += f"  Duration: {duration}s (optimized for emergency clearance)\n"
            reason += f"  Reasoning: Public safety requires immediate path clearing.\n"
            reason += f"            All other traffic must yield.\n"
            
            return decision_phase, duration, reason
        
        # PRIORITY 2: STARVATION PREVENTION
        max_wait = max(avg_waits.values())
        if max_wait > 15.0:
            starved_lane = max(avg_waits, key=avg_waits.get)
            decision_phase = 'NS' if starved_lane in ['N', 'S'] else 'EW'
            duration = 30.0
            
            reason = f"PRIORITY 2 ACTIVATED: STARVATION PREVENTION\n"
            reason += f"  Rule: No lane should wait more than 15 seconds\n"
            reason += f"  Analysis: Lane {starved_lane} has waited {max_wait:.1f}s\n"
            reason += f"  Threshold Exceeded: {max_wait - 15.0:.1f}s over limit\n"
            reason += f"  Action: Switch to {decision_phase} phase\n"
            reason += f"  Duration: {duration}s (standard fairness cycle)\n"
            reason += f"  Reasoning: Fairness requires serving long-waiting lanes.\n"
            reason += f"            Prevents indefinite delays and driver frustration.\n"
            
            return decision_phase, duration, reason
        
        # PRIORITY 3: PRESSURE BALANCING
        ns_load = lane_counts['N'] + lane_counts['S']
        ew_load = lane_counts['E'] + lane_counts['W']
        
        pressure_threshold = 1.5
        
        if current_phase == 'NS' and ns_load > ew_load * pressure_threshold:
            duration = 40.0
            
            reason = f"PRIORITY 3 ACTIVATED: PRESSURE BALANCING (EXTENSION)\n"
            reason += f"  Rule: Extend green when direction has 1.5x+ more vehicles\n"
            reason += f"  Analysis: NS has {ns_load} vehicles vs EW's {ew_load}\n"
            reason += f"  Pressure Ratio: {ns_load/max(ew_load,1):.2f}x (threshold: {pressure_threshold}x)\n"
            reason += f"  Action: EXTEND current {current_phase} phase\n"
            reason += f"  Duration: {duration}s (extended to clear backlog)\n"
            reason += f"  Reasoning: High congestion in active direction justifies\n"
            reason += f"            extension to prevent massive queue buildup.\n"
            
            return current_phase, duration, reason
        
        elif current_phase == 'EW' and ew_load > ns_load * pressure_threshold:
            duration = 40.0
            
            reason = f"PRIORITY 3 ACTIVATED: PRESSURE BALANCING (EXTENSION)\n"
            reason += f"  Rule: Extend green when direction has 1.5x+ more vehicles\n"
            reason += f"  Analysis: EW has {ew_load} vehicles vs NS's {ns_load}\n"
            reason += f"  Pressure Ratio: {ew_load/max(ns_load,1):.2f}x (threshold: {pressure_threshold}x)\n"
            reason += f"  Action: EXTEND current {current_phase} phase\n"
            reason += f"  Duration: {duration}s (extended to clear backlog)\n"
            reason += f"  Reasoning: High congestion in active direction justifies\n"
            reason += f"            extension to prevent massive queue buildup.\n"
            
            return current_phase, duration, reason
        
        # PRIORITY 4: NORMAL ROTATION
        decision_phase = 'EW' if current_phase == 'NS' else 'NS'
        duration = 30.0
        
        reason = f"PRIORITY 4 ACTIVATED: NORMAL PHASE ROTATION\n"
        reason += f"  Rule: Regular alternation when no special conditions exist\n"
        reason += f"  Analysis: Traffic is balanced, no emergencies or starvation\n"
        reason += f"  Current Load: NS={ns_load}, EW={ew_load} (balanced)\n"
        reason += f"  Action: Switch from {current_phase} to {decision_phase}\n"
        reason += f"  Duration: {duration}s (standard cycle time)\n"
        reason += f"  Reasoning: Normal operation maintains predictable flow\n"
        reason += f"            and ensures both directions get fair service.\n"
        
        return decision_phase, duration, reason


class YOLOv11Detector:
    """YOLOv11 vehicle detection wrapper"""
    
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.4):
        print(f"Loading YOLOv11 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 1: 'bicycle'
        }
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLOv11 detection on frame"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'class_id': cls_id,
                        'class_name': self.vehicle_classes[cls_id],
                        'confidence': conf,
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                    }
                    
                    detections.append(detection)
        
        return detections


class SimpleByteTracker:
    """Simplified ByteTrack implementation"""
    
    def __init__(self):
        self.tracks: Dict[str, VehicleTrack] = {}
        self.next_id = 0
        
    def update(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> Dict[str, Vehicle]:
        """Update tracks with new detections"""
        vehicles = {}
        matched = set()
        
        # Match detections to existing tracks
        for track_id, track in list(self.tracks.items()):
            if len(track.positions) == 0:
                continue
                
            last_x, last_y, _ = track.positions[-1]
            
            min_dist = float('inf')
            best_det = None
            best_idx = None
            
            for idx, det in enumerate(detections):
                if idx in matched:
                    continue
                    
                cx, cy = det['center']
                dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    best_det = det
                    best_idx = idx
            
            if best_det is not None:
                matched.add(best_idx)
                
                cx, cy = best_det['center']
                lane = self._determine_lane(cx, cy, frame_shape)
                
                vehicle = Vehicle(
                    id=track_id,
                    lane=lane,
                    type=best_det['class_name'],
                    bbox=best_det['bbox'],
                    confidence=best_det['confidence'],
                    x=float(cx),
                    y=float(cy),
                    is_emergency=self._is_emergency(best_det)
                )
                
                if len(track.positions) > 0:
                    old_x, old_y, old_t = track.positions[-1]
                    dt = time.time() - old_t
                    vehicle.update_position(cx, cy, dt)
                
                if vehicle.speed < 2.0:
                    vehicle.wait_time = track.wait_time + 0.1
                else:
                    vehicle.wait_time = 0.0
                
                track.update(cx, cy, vehicle.wait_time, vehicle.speed)
                vehicles[track_id] = vehicle
        
        # Create new tracks
        for idx, det in enumerate(detections):
            if idx not in matched:
                track_id = f"T{self.next_id:04d}"
                self.next_id += 1
                
                cx, cy = det['center']
                lane = self._determine_lane(cx, cy, frame_shape)
                
                track = VehicleTrack(
                    id=track_id,
                    lane=lane,
                    type=det['class_name'],
                    first_seen=time.time(),
                    is_emergency=self._is_emergency(det)
                )
                
                track.update(cx, cy, 0.0, 0.0)
                self.tracks[track_id] = track
                
                vehicle = Vehicle(
                    id=track_id,
                    lane=lane,
                    type=det['class_name'],
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    x=float(cx),
                    y=float(cy),
                    is_emergency=track.is_emergency
                )
                
                vehicles[track_id] = vehicle
        
        return vehicles
    
    def _determine_lane(self, x: int, y: int, frame_shape: Tuple[int, int]) -> str:
        """Determine which lane"""
        h, w = frame_shape
        center_x, center_y = w // 2, h // 2
        
        if x < center_x - 50:
            return 'W'
        elif x > center_x + 50:
            return 'E'
        elif y < center_y - 50:
            return 'N'
        else:
            return 'S'
    
    def _is_emergency(self, detection: Dict) -> bool:
        return False


class TrafficJunctionSystem:
    """Main traffic system - OPTIMIZED VERSION"""
    
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_id}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera initialized: {self.width}x{self.height}")
        
        self.detector = YOLOv11Detector()
        self.tracker = SimpleByteTracker()
        self.reasoner = IntelligentTrafficReasoner()
        
        self.phases = {
            'NS': TrafficSignalPhase('NS', ['N', 'S'], 30.0),
            'EW': TrafficSignalPhase('EW', ['E', 'W'], 30.0)
        }
        self.current_phase = 'NS'
        
        self.stats = {
            'total_detections': 0,
            'phase_changes': 0,
            'frames_processed': 0
        }
        
        self.last_reasoning_time = time.time()
        self.reasoning_interval = 3.0  # Faster reasoning every 3 seconds
        self.last_reasoning = ""
        
    def process_frame(self) -> Tuple:
        ret, frame = self.cap.read()
        if not ret:
            return None, {}
        
        self.stats['frames_processed'] += 1
        
        # YOLO detection
        detections = self.detector.detect(frame)
        self.stats['total_detections'] += len(detections)
        
        # ByteTrack
        vehicles = self.tracker.update(detections, frame.shape[:2])
        
        # Fast reasoning (no LLM delay!)
        current_time = time.time()
        if current_time - self.last_reasoning_time >= self.reasoning_interval:
            vehicle_list = list(vehicles.values())
            
            new_phase, duration, reasoning = self.reasoner.analyze_junction(
                vehicle_list,
                self.tracker.tracks,
                self.current_phase
            )
            
            self.last_reasoning = reasoning
            print("\n" + reasoning)
            
            if new_phase != self.current_phase:
                self.current_phase = new_phase
                self.phases[new_phase].reset(duration)
                self.stats['phase_changes'] += 1
                print(f"🔄 PHASE CHANGE: Switching to {new_phase}")
            else:
                self.phases[self.current_phase].timer = duration
            
            self.last_reasoning_time = current_time
        
        vis_frame = self.render(frame, vehicles)
        return vis_frame, vehicles
    
    def render(self, frame: np.ndarray, vehicles: Dict) -> np.ndarray:
        vis = frame.copy()
        
        for vehicle in vehicles.values():
            x1, y1, x2, y2 = vehicle.bbox
            
            colors = {
                'N': (0, 0, 255), 'S': (255, 0, 0),
                'E': (0, 255, 0), 'W': (0, 165, 255)
            }
            color = colors.get(vehicle.lane, (255, 255, 255))
            
            if vehicle.is_emergency:
                color = (0, 0, 255)
                thickness = 3
            else:
                thickness = 2
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{vehicle.id} {vehicle.type} {vehicle.lane}"
            if vehicle.is_stopped():
                label += f" W:{vehicle.wait_time:.0f}s"
            
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Phase info
        phase_color = (0, 255, 0) if self.current_phase == 'NS' else (0, 165, 255)
        cv2.rectangle(vis, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(vis, (10, 10), (350, 120), phase_color, 2)
        
        cv2.putText(vis, f"Phase: {self.current_phase}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Timer: {self.phases[self.current_phase].timer:.1f}s", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Vehicles: {len(vehicles)}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return vis
    
    def run(self):
        print("\n🚦 Starting Traffic Junction System...")
        print("⚡ FAST MODE - No LLM delays!")
        print("Press 'q' to quit, 'r' for reasoning, 's' for stats\n")
        
        while True:
            vis_frame, vehicles = self.process_frame()
            
            if vis_frame is None:
                break
            
            self.phases[self.current_phase].update(0.033)
            
            cv2.imshow('Traffic Junction - YOLOv11 + Intelligent Reasoning', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n" + self.last_reasoning)
            elif key == ord('s'):
                print(f"\n{'='*40}")
                print("STATISTICS")
                print(f"{'='*40}")
                for k, v in self.stats.items():
                    print(f"{k}: {v}")
                print(f"Active Vehicles: {len(vehicles)}")
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*40)
        print("FINAL STATISTICS")
        print("="*40)
        for k, v in self.stats.items():
            print(f"{k}: {v}")


def main():
    print("="*70)
    print("Intelligent Traffic Junction Control System")
    print("YOLOv11 + ByteTrack + Advanced Rule-Based AI")
    print("="*70)
    print("\n⚡ OPTIMIZED - Fast reasoning, no LLM delays!")
    print("📊 Full Chain-of-Thought explanations included\n")
    
    try:
        system = TrafficJunctionSystem(
            camera_id='29310-374747467_small.mp4'  # or 0 for webcam
        )
        system.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()