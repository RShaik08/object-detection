"""
Intelligent Traffic Junction Control System
YOLOv11 + ByteTrack + Advanced Rule-Based Reasoning
+ DEEP REINFORCEMENT LEARNING (DQN) for adaptive signal control

OPTIMIZED FOR PERFORMANCE - No LLM delays!
Supports both rule-based and RL modes for research comparison.

Requirements:
pip install opencv-python numpy ultralytics torch
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import random
from collections import deque
import math

# YOLOv11 for real-time detection
from ultralytics import YOLO

# ===== RL ADDITIONS =====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------------------- Existing Data Classes ----------------------------
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
        self.vx = (new_x - self.x) / max(dt, 0.001)
        self.vy = (new_y - self.y) / max(dt, 0.001)
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        self.x = new_x
        self.y = new_y

    def is_stopped(self, threshold: float = 2.0) -> bool:
        return self.speed < threshold

    def get_center(self) -> Tuple[int, int]:
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

# ---------------------------- Rule-Based Reasoner (unchanged) ----------------------------
class IntelligentTrafficReasoner:
    """Advanced Rule-Based Reasoning Engine (original)"""
    def __init__(self):
        self.reasoning_log = []
        self.decision_count = 0

    def analyze_junction(self, vehicles: List[Vehicle], tracks: Dict[str, VehicleTrack],
                        current_phase: str) -> Tuple[str, float, str]:
        self.decision_count += 1
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
                emergency_distance = min(emergency_distance, abs(640 - v.x) + abs(360 - v.y))

        avg_waits = {}
        avg_speeds = {}
        for lane in ['N', 'S', 'E', 'W']:
            times = lane_wait_times[lane]
            speeds = lane_speeds[lane]
            avg_waits[lane] = sum(times) / len(times) if times else 0.0
            avg_speeds[lane] = sum(speeds) / len(speeds) if speeds else 0.0

        reasoning = self._build_detailed_reasoning(
            current_phase, lane_counts, avg_waits, avg_speeds,
            emergency_detected, emergency_lane, emergency_distance, len(vehicles)
        )
        decision_phase, decision_duration, decision_reason = self._make_decision(
            current_phase, lane_counts, avg_waits, avg_speeds,
            emergency_detected, emergency_lane, emergency_distance
        )
        full_reasoning = reasoning + "\n" + decision_reason
        full_reasoning += f"\n\n{'='*60}\nFINAL DECISION: Phase={decision_phase}, Duration={decision_duration}s\n{'='*60}\n"
        self.reasoning_log.append(full_reasoning)
        return decision_phase, decision_duration, full_reasoning

    def _build_detailed_reasoning(self, current_phase: str, lane_counts: Dict,
                                  avg_waits: Dict, avg_speeds: Dict,
                                  emergency: bool, emergency_lane: str,
                                  emergency_dist: float, total_vehicles: int) -> str:
        reasoning = f"{'='*60}\nTRAFFIC-R1 INTELLIGENT REASONING ENGINE\n{'='*60}\n"
        reasoning += f"Decision #{self.decision_count} | Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
        reasoning += f"CURRENT STATE ANALYSIS:\n  Active Phase: {current_phase}\n  Total Vehicles: {total_vehicles}\n\n"
        reasoning += f"LANE-BY-LANE BREAKDOWN:\n"
        for lane in ['N', 'S', 'E', 'W']:
            reasoning += f"  Lane {lane}:\n    • Vehicles: {lane_counts[lane]}\n    • Avg Wait Time: {avg_waits[lane]:.1f}s\n    • Avg Speed: {avg_speeds[lane]:.1f} px/s\n"
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
        reasoning += f"  NS Direction: {ns_load} vehicles\n  EW Direction: {ew_load} vehicles\n"
        if emergency:
            reasoning += f"\n🚨 EMERGENCY ALERT: Lane {emergency_lane} (dist {emergency_dist:.0f}px)\n"
        return reasoning

    def _make_decision(self, current_phase: str, lane_counts: Dict,
                      avg_waits: Dict, avg_speeds: Dict,
                      emergency: bool, emergency_lane: str,
                      emergency_dist: float) -> Tuple[str, float, str]:
        if emergency:
            decision_phase = 'NS' if emergency_lane in ['N','S'] else 'EW'
            duration = 15.0 if emergency_dist < 200 else 10.0
            reason = f"PRIORITY 1: EMERGENCY → {decision_phase} for {duration}s"
            return decision_phase, duration, reason
        max_wait = max(avg_waits.values())
        if max_wait > 15.0:
            starved_lane = max(avg_waits, key=avg_waits.get)
            decision_phase = 'NS' if starved_lane in ['N','S'] else 'EW'
            reason = f"PRIORITY 2: STARVATION (lane {starved_lane} waited {max_wait:.1f}s) → {decision_phase}"
            return decision_phase, 30.0, reason
        ns_load = lane_counts['N'] + lane_counts['S']
        ew_load = lane_counts['E'] + lane_counts['W']
        pressure_threshold = 1.5
        if current_phase == 'NS' and ns_load > ew_load * pressure_threshold:
            reason = f"PRIORITY 3: PRESSURE (NS {ns_load} > {ew_load}*1.5) → EXTEND NS"
            return current_phase, 40.0, reason
        if current_phase == 'EW' and ew_load > ns_load * pressure_threshold:
            reason = f"PRIORITY 3: PRESSURE (EW {ew_load} > {ns_load}*1.5) → EXTEND EW"
            return current_phase, 40.0, reason
        decision_phase = 'EW' if current_phase == 'NS' else 'NS'
        reason = f"PRIORITY 4: NORMAL ROTATION → {decision_phase}"
        return decision_phase, 30.0, reason

# ---------------------------- YOLOv11 Detector (unchanged) ----------------------------
class YOLOv11Detector:
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.4):
        print(f"Loading YOLOv11 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 1: 'bicycle'}

    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'class_id': cls_id,
                        'class_name': self.vehicle_classes[cls_id],
                        'confidence': conf,
                        'center': ((int(x1)+int(x2))//2, (int(y1)+int(y2))//2)
                    })
        return detections

# ---------------------------- Simple ByteTracker (unchanged) ----------------------------
class SimpleByteTracker:
    def __init__(self):
        self.tracks: Dict[str, VehicleTrack] = {}
        self.next_id = 0

    def update(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> Dict[str, Vehicle]:
        vehicles = {}
        matched = set()
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
                    id=track_id, lane=lane, type=best_det['class_name'],
                    bbox=best_det['bbox'], confidence=best_det['confidence'],
                    x=float(cx), y=float(cy), is_emergency=self._is_emergency(best_det)
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
        for idx, det in enumerate(detections):
            if idx not in matched:
                track_id = f"T{self.next_id:04d}"
                self.next_id += 1
                cx, cy = det['center']
                lane = self._determine_lane(cx, cy, frame_shape)
                track = VehicleTrack(
                    id=track_id, lane=lane, type=det['class_name'],
                    first_seen=time.time(), is_emergency=self._is_emergency(det)
                )
                track.update(cx, cy, 0.0, 0.0)
                self.tracks[track_id] = track
                vehicle = Vehicle(
                    id=track_id, lane=lane, type=det['class_name'],
                    bbox=det['bbox'], confidence=det['confidence'],
                    x=float(cx), y=float(cy), is_emergency=track.is_emergency
                )
                vehicles[track_id] = vehicle
        return vehicles

    def _determine_lane(self, x: int, y: int, frame_shape: Tuple[int, int]) -> str:
        h, w = frame_shape
        cx, cy = w//2, h//2
        if x < cx - 50: return 'W'
        elif x > cx + 50: return 'E'
        elif y < cy - 50: return 'N'
        else: return 'S'

    def _is_emergency(self, detection: Dict) -> bool:
        # Simple heuristic: 10% of buses/trucks are emergency vehicles
        if detection['class_name'] in ['bus', 'truck']:
            return random.random() < 0.1
        return False

# ============================= RL ADDITIONS =============================

class DQN(nn.Module):
    """Deep Q-Network for traffic signal control."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class RLTrafficAgent:
    """
    DQN agent that chooses (phase, duration) actions.
    Action space: 8 discrete actions
        0: NS, short (15s)      1: NS, medium (25s)    2: NS, long (35s)    3: NS, extra-long (45s)
        4: EW, short (15s)      5: EW, medium (25s)    6: EW, long (35s)    7: EW, extra-long (45s)
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 8,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=20000)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax().item())

    def get_action_duration(self, action: int) -> Tuple[str, float]:
        """Map action index to (phase, duration)."""
        durations = [15, 25, 35, 45]
        if action < 4:
            return 'NS', durations[action]
        else:
            return 'EW', durations[action - 4]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


def extract_state(vehicles: Dict[str, Vehicle], current_phase: str) -> np.ndarray:
    """
    Build state vector:
    - Vehicle count per lane (N, S, E, W)
    - Average waiting time per lane (stopped vehicles)
    - Queue length per lane (number of stopped vehicles)
    - Current phase one-hot (NS, EW)
    """
    counts = {'N':0, 'S':0, 'E':0, 'W':0}
    wait_times = {'N':[], 'S':[], 'E':[], 'W':[]}
    queue_lengths = {'N':0, 'S':0, 'E':0, 'W':0}
    for v in vehicles.values():
        counts[v.lane] += 1
        if v.is_stopped():
            wait_times[v.lane].append(v.wait_time)
            queue_lengths[v.lane] += 1
    avg_waits = [np.mean(wait_times[l]) if wait_times[l] else 0.0 for l in ['N','S','E','W']]
    state = [
        counts['N'], counts['S'], counts['E'], counts['W'],
        avg_waits[0], avg_waits[1], avg_waits[2], avg_waits[3],
        queue_lengths['N'], queue_lengths['S'], queue_lengths['E'], queue_lengths['W'],
        1.0 if current_phase == 'NS' else 0.0,
        1.0 if current_phase == 'EW' else 0.0
    ]
    return np.array(state, dtype=np.float32)


def compute_reward(vehicles: Dict[str, Vehicle],
                   alpha: float = 0.5,
                   beta: float = 2.0) -> float:
    """
    Reward = - (total_wait_time) - alpha * imbalance - beta * starvation_penalty
    where imbalance = |NS_vehicles - EW_vehicles|,
    starvation = max(0, max_wait - 15)
    """
    total_wait = sum(v.wait_time for v in vehicles.values())
    ns_count = sum(1 for v in vehicles.values() if v.lane in ['N','S'])
    ew_count = sum(1 for v in vehicles.values() if v.lane in ['E','W'])
    imbalance = abs(ns_count - ew_count)
    max_wait = max([v.wait_time for v in vehicles.values()], default=0.0)
    starvation_penalty = max(0.0, max_wait - 15.0)
    reward = -total_wait - alpha * imbalance - beta * starvation_penalty
    return reward


# ---------------------------- Main Traffic System (upgraded) ----------------------------
class TrafficJunctionSystem:
    """
    Main system. Supports two modes:
        mode='rule'   : original rule-based reasoner
        mode='rl'     : DQN agent (train or evaluate)
    """
    def __init__(self, camera_id, mode: str = 'rule', rl_train: bool = False, rl_model_path: str = None):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_id}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera initialized: {self.width}x{self.height}")

        self.detector = YOLOv11Detector()
        self.tracker = SimpleByteTracker()
        self.mode = mode
        self.rl_train = rl_train

        self.phases = {
            'NS': TrafficSignalPhase('NS', ['N', 'S'], 30.0),
            'EW': TrafficSignalPhase('EW', ['E', 'W'], 30.0)
        }
        self.current_phase = 'NS'

        self.stats = {
            'total_detections': 0,
            'phase_changes': 0,
            'frames_processed': 0,
            'total_reward': 0.0,        # only for RL
            'avg_wait_time': 0.0,
            'max_wait_time': 0.0
        }

        self.last_reasoning_time = time.time()
        self.reasoning_interval = 3.0   # decision every 3 seconds
        self.last_reasoning = ""

        # Rule-based reasoner (always available)
        self.rule_reasoner = IntelligentTrafficReasoner()

        # RL agent
        if mode == 'rl':
            state_dim = 4 + 4 + 4 + 2  # counts(4) + avg_waits(4) + queues(4) + phase_onehot(2)
            self.rl_agent = RLTrafficAgent(state_dim=state_dim, action_dim=8)
            if rl_model_path and not rl_train:
                self.rl_agent.load(rl_model_path)
                print(f"Loaded RL model from {rl_model_path}")
            self.last_state = None
            self.last_action = None
            self.episode_done = False
            print("RL mode active. Agent ready.")

    def process_frame(self) -> Tuple[Optional[np.ndarray], Dict]:
        ret, frame = self.cap.read()
        if not ret:
            return None, {}

        self.stats['frames_processed'] += 1

        # Detection and tracking
        detections = self.detector.detect(frame)
        self.stats['total_detections'] += len(detections)
        vehicles = self.tracker.update(detections, frame.shape[:2])

        current_time = time.time()
        if current_time - self.last_reasoning_time >= self.reasoning_interval:
            vehicle_list = list(vehicles.values())

            if self.mode == 'rule':
                # Original rule-based decision
                new_phase, duration, reasoning = self.rule_reasoner.analyze_junction(
                    vehicle_list, self.tracker.tracks, self.current_phase
                )
                self.last_reasoning = reasoning
                print("\n" + reasoning)
                self._apply_decision(new_phase, duration)

            else:  # RL mode
                # Extract current state
                state = extract_state(vehicles, self.current_phase)

                # For training, we need to compute reward from previous step
                if self.rl_train and self.last_state is not None:
                    reward = compute_reward(vehicles)
                    self.stats['total_reward'] += reward
                    self.rl_agent.store_transition(self.last_state, self.last_action,
                                                   reward, state, False)
                    self.rl_agent.train(batch_size=64)

                # Select action (epsilon-greedy only if training)
                action = self.rl_agent.select_action(state, eval_mode=not self.rl_train)
                new_phase, duration = self.rl_agent.get_action_duration(action)

                # Log RL decision
                reasoning = f"RL DECISION: Phase={new_phase}, Duration={duration}s, Action={action}"
                self.last_reasoning = reasoning
                print(f"\n{reasoning}")

                # Remember for next step
                self.last_state = state
                self.last_action = action

                self._apply_decision(new_phase, duration)

            self.last_reasoning_time = current_time

        # Update timer
        self.phases[self.current_phase].update(0.033)  # ~30 fps

        # Update statistics for wait times
        if vehicles:
            waits = [v.wait_time for v in vehicles.values()]
            self.stats['avg_wait_time'] = np.mean(waits)
            self.stats['max_wait_time'] = max(waits)

        vis_frame = self.render(frame, vehicles)
        return vis_frame, vehicles

    def _apply_decision(self, new_phase: str, duration: float):
        """Apply the decision (phase change or extension)."""
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.phases[new_phase].reset(duration)
            self.stats['phase_changes'] += 1
            print(f"🔄 PHASE CHANGE: Switching to {new_phase} (duration {duration:.1f}s)")
        else:
            self.phases[self.current_phase].timer = duration
            print(f"⏱️ EXTEND {self.current_phase} to {duration:.1f}s")

    def render(self, frame: np.ndarray, vehicles: Dict) -> np.ndarray:
        vis = frame.copy()
        overlay = vis.copy()
        height, width = vis.shape[:2]

        colors = {'N': (0,0,255), 'S': (255,0,255), 'E': (0,255,0), 'W': (255,255,0)}
        cv2.putText(vis, "N", (width//2-10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,colors['N'],2)
        cv2.putText(vis, "S", (width//2-10,height-30), cv2.FONT_HERSHEY_SIMPLEX,0.7,colors['S'],2)
        cv2.putText(vis, "E", (width-30,height//2+10), cv2.FONT_HERSHEY_SIMPLEX,0.7,colors['E'],2)
        cv2.putText(vis, "W", (30,height//2+10), cv2.FONT_HERSHEY_SIMPLEX,0.7,colors['W'],2)

        for vehicle in vehicles.values():
            x1,y1,x2,y2 = vehicle.bbox
            color = colors.get(vehicle.lane, (255,255,255))
            if vehicle.is_emergency:
                color = (0,0,255)
                thickness = 4
                if int(time.time()*4)%2:
                    cv2.rectangle(overlay, (x1-5,y1-5), (x2+5,y2+5), (0,0,255), -1)
            else:
                thickness = 2
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
            label = f"{vehicle.id} {vehicle.type} {vehicle.lane}"
            if vehicle.is_stopped():
                label += f" W:{vehicle.wait_time:.0f}s"
            (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(vis, (x1,y1-h-10), (x1+w,y1), (0,0,0), -1)
            cv2.putText(vis, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        if any(v.is_emergency for v in vehicles.values()):
            cv2.addWeighted(overlay,0.3,vis,0.7,0,vis)

        # Info panel
        phase_color = (0,255,0) if self.current_phase=='NS' else (0,165,255)
        cv2.rectangle(vis, (10,10), (350,150), (50,50,50), -1)
        cv2.rectangle(vis, (10,10), (350,150), phase_color, 2)
        cv2.putText(vis, f"Mode: {self.mode.upper()}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(vis, f"Phase: {self.current_phase}", (20,70), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(vis, f"Timer: {self.phases[self.current_phase].timer:.1f}s", (20,100), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(vis, f"Vehicles: {len(vehicles)}", (20,130), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # Lane status
        y_offset = 170
        cv2.rectangle(vis, (10,y_offset), (200,y_offset+100), (50,50,50), -1)
        cv2.rectangle(vis, (10,y_offset), (200,y_offset+100), (100,100,100), 2)
        cv2.putText(vis, "LANE STATUS", (20,y_offset+25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        for i, lane in enumerate(['N','S','E','W']):
            count = sum(1 for v in vehicles.values() if v.lane == lane)
            y = y_offset+45 + i*20
            cv2.putText(vis, f"{lane}: {count:2d}", (20,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,colors[lane],2)

        return vis

    def run(self):
        print(f"\n🚦 Starting Traffic Junction System (mode={self.mode})")
        print("⚡ FAST MODE - No LLM delays!")
        print("Press 'q' to quit, 'r' for reasoning, 's' for stats")
        print("Press 'e' to SIMULATE EMERGENCY VEHICLE\n")

        emergency_mode = False

        while True:
            vis_frame, vehicles = self.process_frame()
            if vis_frame is None:
                break

            cv2.imshow('Traffic Junction - YOLOv11 + AI', vis_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n" + self.last_reasoning)
            elif key == ord('s'):
                print(f"\n{'='*40}\nSTATISTICS\n{'='*40}")
                for k,v in self.stats.items():
                    print(f"{k}: {v}")
                print(f"Active Vehicles: {len(vehicles)}")
            elif key == ord('e'):
                emergency_mode = not emergency_mode
                print(f"🚨 Emergency mode: {'ON' if emergency_mode else 'OFF'}")
                # Mark first vehicle as emergency for demo
                if emergency_mode and len(vehicles)>0:
                    list(vehicles.values())[0].is_emergency = True

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*40)
        print("FINAL STATISTICS")
        print("="*40)
        for k,v in self.stats.items():
            print(f"{k}: {v}")
        if self.mode == 'rl' and self.rl_train:
            # Save trained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"rl_traffic_model_{timestamp}.pt"
            self.rl_agent.save(model_path)
            print(f"RL model saved to {model_path}")

def export_results(system):
    """Export statistics and reasoning logs for paper graphs."""
    import csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Stats CSV
    with open(f'results_stats_{system.mode}_{timestamp}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in system.stats.items():
            writer.writerow([key, value])
        writer.writerow(['Total Vehicles Tracked', len(system.tracker.tracks)])
        if system.mode == 'rule':
            writer.writerow(['Reasoning Calls', system.rule_reasoner.decision_count])
        else:
            writer.writerow(['RL Training Steps', system.rl_agent.steps if hasattr(system, 'rl_agent') else 0])
    # Reasoning log (only for rule mode)
    if system.mode == 'rule':
        with open(f'reasoning_log_{timestamp}.txt', 'w', encoding='utf-8') as f:
            for reasoning in system.rule_reasoner.reasoning_log:
                f.write(reasoning)
                f.write("\n" + "="*80 + "\n\n")
        print(f"✓ Reasoning log saved to reasoning_log_{timestamp}.txt")
    print(f"✓ Results exported to results_stats_{system.mode}_{timestamp}.csv")

def main():
    print("="*70)
    print("Intelligent Traffic Junction Control System")
    print("YOLOv11 + ByteTrack + Rule-Based AI + DQN (RL)")
    print("="*70)
    print("\n⚡ OPTIMIZED - Fast reasoning, no LLM delays!")
    print("📊 Full Chain-of-Thought explanations (rule mode)")
    print("🧠 DQN agent learns optimal signal timing (RL mode)\n")

    # Choose mode: 'rule' or 'rl'
    mode = input("Select mode (rule/rl) [default=rule]: ").strip().lower() or 'rule'
    rl_train = False
    rl_model_path = None
    if mode == 'rl':
        train_or_eval = input("Train new model (train) or evaluate existing (eval)? [default=train]: ").strip().lower()
        rl_train = (train_or_eval != 'eval')
        if not rl_train:
            rl_model_path = input("Path to saved RL model (.pt): ").strip()
            if not rl_model_path:
                print("No model provided, switching to train mode.")
                rl_train = True

    try:
        # Update camera path as needed
        camera_path = r'C:\Users\shaik\crossroad\videos\29310-374747467_small.mp4'
        system = TrafficJunctionSystem(
            camera_id=camera_path,
            mode=mode,
            rl_train=rl_train,
            rl_model_path=rl_model_path
        )
        system.run()
        export_results(system)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
