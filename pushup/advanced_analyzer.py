

import cv2
import numpy as np
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class RepMetrics:
    """Store metrics for each repetition"""
    rep_number: int
    start_time: Optional[float]
    end_time: Optional[float]
    duration: float
    elbow_angle: float
    shoulder_angle: float
    hip_angle: float
    depth: float  # minimum elbow angle reached
    form_quality: float  # 0-100
    is_correct: bool
    feedback: str
    max_hip_sag: float  # deviation from straight line
    symmetry_score: float  # left-right balance
    # No cheat detection flags in this dataclass (kept focused on rep metrics)


class PushUpAnalyzer:
    """Advanced push-up form analyzer with comprehensive metrics"""
    
    def __init__(self, history_length=30):
        self.rep_history: List[RepMetrics] = []
        self.angle_history = deque(maxlen=history_length)
        self.depth_history = deque(maxlen=history_length)
        self.time_history = deque(maxlen=history_length)
        
        # Thresholds
        self.DOWN_THRESHOLD = 100
        self.UP_THRESHOLD = 140
        self.GOOD_DEPTH = 85  # degrees
        self.MIN_DEPTH = 60   # degrees
        
        # Current state
        self.stage = 'up'
        self.counter = 0
        self.rep_start_time = None
        self.rep_angles = []
        self.rep_shoulder_angles = []
        self.rep_hip_angles = []
        self.min_depth_this_rep = 180
        self.max_hip_sag = 0
        
        # Fatigue tracking
        self.fatigue_score = 0
        self.fatigue_onset_time = None
        
    def analyze_body_alignment(self, pts_body, pts_aligned):
        """
        Analyze alignment of body (hip, shoulder, ankle)
        Returns alignment score 0-100
        """
        try:
            if pts_body is None or len(pts_body) < 6:
                return 100
            
            # Extract key points
            shoulder = np.array(pts_body[1], dtype=np.float32)  # shoulder
            hip = np.array(pts_body[5], dtype=np.float32)       # hip
            ankle = np.array(pts_body[6], dtype=np.float32) if len(pts_body) > 6 else np.array(pts_body[11], dtype=np.float32)  # ankle or knee
            
            # Calculate if body forms straight line
            # Vector from ankle to shoulder
            v_shoulder = shoulder - ankle
            v_hip = hip - ankle
            
            shoulder_norm = np.linalg.norm(v_shoulder)
            if shoulder_norm < 1:
                return 100
            
            # Project hip onto line from ankle to shoulder
            v_shoulder_unit = v_shoulder / shoulder_norm
            proj_length = np.dot(v_hip, v_shoulder_unit)
            projection = ankle + proj_length * v_shoulder_unit
            
            # Distance from hip to projected point (sag/arch)
            sag = np.linalg.norm(v_hip - (proj_length * v_shoulder_unit))
            
            # Normalize sag (0-100 scale)
            alignment_score = max(0, 100 - (sag / 50) * 100)  # 50px = worst score
            return float(min(100, alignment_score))
        except Exception as e:
            print(f"Alignment calculation error: {e}")
            return 100
    
    def calculate_hip_sag(self, pts_body):
        """Calculate hip sagging/rising relative to shoulder-ankle line"""
        try:
            if pts_body is None or len(pts_body) < 6:
                return 0
            
            shoulder = np.array(pts_body[1], dtype=np.float32)
            hip = np.array(pts_body[5], dtype=np.float32)
            ankle = np.array(pts_body[6], dtype=np.float32) if len(pts_body) > 6 else np.array(pts_body[11], dtype=np.float32)
            
            # Line from shoulder to ankle
            v = ankle - shoulder
            v_len = np.linalg.norm(v)
            
            if v_len < 1:
                return 0
            
            v_unit = v / v_len
            
            # Vector from shoulder to hip
            hip_v = hip - shoulder
            
            # Perpendicular distance (positive = sag down, negative = raised)
            perp_dist = abs(hip_v[0] * (-v_unit[1]) + hip_v[1] * v_unit[0])
            return float(perp_dist)
        except Exception as e:
            print(f"Hip sag calculation error: {e}")
            return 0
    
    def calculate_symmetry(self, left_elbow_angle, right_elbow_angle):
        """Calculate symmetry between left and right sides (0-100)"""
        try:
            if left_elbow_angle is None or right_elbow_angle is None:
                return 100.0
            
            left_val = float(left_elbow_angle) if not isinstance(left_elbow_angle, (list, np.ndarray)) else float(left_elbow_angle[0]) if len(left_elbow_angle) > 0 else 0
            right_val = float(right_elbow_angle) if not isinstance(right_elbow_angle, (list, np.ndarray)) else float(right_elbow_angle[0]) if len(right_elbow_angle) > 0 else 0
            
            if left_val == 0 and right_val == 0:
                return 100.0
            
            angle_diff = abs(left_val - right_val)
            symmetry = max(0, 100 - angle_diff * 2)  # 25° diff = 50 score
            return float(min(100, symmetry))
        except Exception as e:
            print(f"Symmetry calculation error: {e}")
            return 100.0
    
    def generate_feedback(self, elbow_angle, hip_sag, alignment_score, depth, 
                         symmetry, stage):
        """Generate real-time feedback"""
        feedback_list = []
        
        if stage == 'down':
            if elbow_angle > self.UP_THRESHOLD:
                feedback_list.append("Go deeper")
            if hip_sag > 40:
                feedback_list.append("Don't sag hips")
            elif hip_sag > 25:
                feedback_list.append("Engage core")
            if symmetry < 80:
                feedback_list.append("Balance sides")
            if alignment_score < 70:
                feedback_list.append("Straighten back")
        
        elif stage == 'up':
            if hip_sag > 50:
                feedback_list.append("Raise hips")
            if symmetry < 75:
                feedback_list.append("Keep balanced")
        
        return " | ".join(feedback_list) if feedback_list else "Good form!"
    
    def calculate_form_quality(self, depth, alignment, symmetry, hip_sag):
        """
        Calculate overall form quality (0-100)
        Weighted components:
        - Depth (40%): how deep the push-up goes
        - Alignment (30%): body straightness
        - Symmetry (20%): left-right balance
        - Stability (10%): minimal hip sag
        """
        depth_score = min(100, (depth / self.GOOD_DEPTH) * 100) if depth < self.GOOD_DEPTH else 100
        
        stability_score = max(0, 100 - (hip_sag / 30) * 100)
        
        quality = (depth_score * 0.40 + 
                  alignment * 0.30 + 
                  symmetry * 0.20 + 
                  stability_score * 0.10)
        
        return min(100, max(0, quality))
    
    def update(self, elbow_angle, shoulder_angle, hip_angle, 
               left_elbow=None, right_elbow=None, pts_body=None):
        """Update analyzer with current frame data"""
        try:
            # Convert angles to float if they're arrays
            elbow_angle = float(elbow_angle) if not isinstance(elbow_angle, float) else elbow_angle
            shoulder_angle = float(shoulder_angle) if not isinstance(shoulder_angle, float) else shoulder_angle
            hip_angle = float(hip_angle) if not isinstance(hip_angle, float) else hip_angle
            
            self.angle_history.append(elbow_angle)
            
            # Calculate metrics
            hip_sag = self.calculate_hip_sag(pts_body) if pts_body is not None else 0
            alignment = self.analyze_body_alignment(pts_body, None) if pts_body is not None else 100
            symmetry = self.calculate_symmetry(left_elbow, right_elbow)
            
            # Rep detection with hysteresis
            if elbow_angle < self.DOWN_THRESHOLD and self.stage == 'up':
                self.stage = 'down'
                self.rep_start_time = time.time()
                self.rep_angles = [elbow_angle]
                self.min_depth_this_rep = elbow_angle
                self.max_hip_sag = 0
                print(f"DOWN detected | Angle: {elbow_angle}°")
            
            elif elbow_angle > self.UP_THRESHOLD and self.stage == 'down':
                self.stage = 'up'
                self.counter += 1
                
                # Complete rep metrics
                rep_duration = time.time() - self.rep_start_time if self.rep_start_time else 0
                self.time_history.append(rep_duration)
                
                depth_score = self.min_depth_this_rep
                form_quality = self.calculate_form_quality(
                    180 - depth_score, alignment, symmetry, self.max_hip_sag
                )
                
                is_correct = (depth_score < self.GOOD_DEPTH and 
                             form_quality > 70 and 
                             rep_duration > 0.5)
                
                rep = RepMetrics(
                    rep_number=self.counter,
                    start_time=self.rep_start_time,
                    end_time=time.time(),
                    duration=rep_duration,
                    elbow_angle=float(elbow_angle),
                    shoulder_angle=float(shoulder_angle),
                    hip_angle=float(hip_angle),
                    depth=float(180 - depth_score),
                    form_quality=float(form_quality),
                    is_correct=is_correct,
                    feedback=self.generate_feedback(elbow_angle, hip_sag, alignment, 
                                                   180 - depth_score, symmetry, 'up'),
                    max_hip_sag=float(self.max_hip_sag),
                    symmetry_score=float(symmetry),
                )
                self.rep_history.append(rep)
                self.depth_history.append(rep.depth)
                
                print(f"✓ REP {self.counter} | Depth: {rep.depth:.0f}° | "
                      f"Quality: {form_quality:.0f}/100 | Time: {rep_duration:.2f}s")
                
                self.rep_start_time = None
            
            # Track current rep data
            if self.stage == 'down':
                self.rep_angles.append(elbow_angle)
                self.min_depth_this_rep = min(self.min_depth_this_rep, elbow_angle)
                self.max_hip_sag = max(self.max_hip_sag, hip_sag)
            
            # Update fatigue detection
            self._update_fatigue()
            
            current_depth = 180 - self.min_depth_this_rep if self.stage == 'down' else 0
            current_form = self.calculate_form_quality(current_depth, alignment, symmetry, hip_sag)
            
            return {
                'stage': self.stage,
                'counter': self.counter,
                'elbow_angle': float(elbow_angle),
                'shoulder_angle': float(shoulder_angle),
                'hip_angle': float(hip_angle),
                'alignment': float(alignment),
                'symmetry': float(symmetry),
                'hip_sag': float(hip_sag),
                'form_quality': float(current_form),
                'feedback': self.generate_feedback(elbow_angle, hip_sag, alignment, 
                                                  current_depth, symmetry, self.stage)
            }
        except Exception as e:
            print(f"Update error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'stage': self.stage,
                'counter': self.counter,
                'elbow_angle': 0,
                'shoulder_angle': 0,
                'hip_angle': 0,
                'alignment': 100,
                'symmetry': 100,
                'hip_sag': 0,
                'form_quality': 0,
                'feedback': 'Error in analysis'
            }
    
    def _update_fatigue(self):
        """Detect fatigue based on tempo and form degradation"""
        if len(self.time_history) < 5:
            return
        
        recent_times = list(self.time_history)[-5:]
        recent_depths = list(self.depth_history)[-5:]
        
        avg_time = np.mean(recent_times)
        avg_depth = np.mean(recent_depths)
        
        # Fatigue indicators
        slowing = avg_time > 1.5
        shallowing = avg_depth < self.GOOD_DEPTH * 0.8
        
        if slowing or shallowing:
            self.fatigue_score = min(10, self.fatigue_score + 1)
            if self.fatigue_onset_time is None:
                self.fatigue_onset_time = time.time()
        else:
            self.fatigue_score = max(0, self.fatigue_score - 0.5)
    
    def get_statistics(self):
        """Generate comprehensive statistics"""
        if not self.rep_history:
            return {}
        
        correct_reps = sum(1 for r in self.rep_history if r.is_correct)
        incorrect_reps = len(self.rep_history) - correct_reps
        
        avg_rep_time = np.mean([r.duration for r in self.rep_history])
        avg_form_quality = np.mean([r.form_quality for r in self.rep_history])
        avg_symmetry = np.mean([r.symmetry_score for r in self.rep_history])
        
        return {
            'total_reps': self.counter,
            'correct_reps': correct_reps,
            'incorrect_reps': incorrect_reps,
            'accuracy_percent': (correct_reps / self.counter * 100) if self.counter > 0 else 0,
            'avg_rep_time': avg_rep_time,
            'avg_form_quality': avg_form_quality,
            'avg_symmetry': avg_symmetry,
            'avg_depth': np.mean([r.depth for r in self.rep_history]),
            'fatigue_score': self.fatigue_score,
            'fatigue_onset_rep': next((r.rep_number for r in self.rep_history 
                                      if r.duration > 2.0), None),
            'best_rep': max(self.rep_history, key=lambda r: r.form_quality) if self.rep_history else None,
            'worst_rep': min(self.rep_history, key=lambda r: r.form_quality) if self.rep_history else None,
        }


def print_advanced_metrics(analyzer):
    """Print comprehensive metrics to terminal (console)."""
    stats = analyzer.get_statistics()
    if not stats:
        print("No reps yet.")
        return

    best = stats.get('best_rep')
    worst = stats.get('worst_rep')

    print("\n=== SESSION METRICS ===")
    print(f"Total reps: {stats.get('total_reps', 0)}")
    print(f"Correct reps: {stats.get('correct_reps', 0)} | Incorrect reps: {stats.get('incorrect_reps', 0)}")
    print(f"Accuracy: {stats.get('accuracy_percent', 0):.1f}%")
    print(f"Avg rep time: {stats.get('avg_rep_time', 0):.2f}s")
    print(f"Avg form quality: {stats.get('avg_form_quality', 0):.1f}/100")
    print(f"Avg symmetry: {stats.get('avg_symmetry', 0):.1f}/100")
    print(f"Avg depth: {stats.get('avg_depth', 0):.1f}°")
    print(f"Fatigue score: {stats.get('fatigue_score', 0)} / 10")
    if stats.get('fatigue_onset_rep'):
        print(f"Fatigue onset rep: {stats.get('fatigue_onset_rep')}")
    if best:
        print(f"Best rep: #{best.rep_number} - Quality: {best.form_quality:.1f}")
    if worst:
        print(f"Worst rep: #{worst.rep_number} - Quality: {worst.form_quality:.1f}")
    print("=======================\n")


def print_athlete_performance(analyzer):
    """Compute and print an overall athlete performance summary for the session.

    This combines accuracy, average form quality, symmetry, consistency and
    endurance (fatigue) into a single 0-100 performance score and prints
    actionable recommendations.
    """
    stats = analyzer.get_statistics()
    if not stats:
        print("No reps yet. Athlete performance unavailable.")
        return

    accuracy = stats.get('accuracy_percent', 0)
    avg_quality = stats.get('avg_form_quality', 0)
    symmetry = stats.get('avg_symmetry', 0)
    fatigue = stats.get('fatigue_score', 0)
    avg_depth = stats.get('avg_depth', 0)

    # Consistency: lower stddev of rep times is better
    durations = [r.duration for r in analyzer.rep_history] if analyzer.rep_history else []
    if durations and len(durations) > 1:
        dur_std = float(np.std(durations))
        dur_mean = float(np.mean(durations)) if np.mean(durations) > 0 else 1.0
        consistency = max(0.0, 100.0 - (dur_std / dur_mean) * 100.0)  # 100 = perfectly consistent
    else:
        consistency = 100.0

    # Endurance score derived from fatigue (0-10). Lower fatigue maps to higher endurance
    endurance = max(0.0, 100.0 - fatigue * 10.0)

    # Weighted aggregation (sum to 100)
    overall = (
        accuracy * 0.35 +
        avg_quality * 0.30 +
        symmetry * 0.15 +
        consistency * 0.10 +
        endurance * 0.10
    )

    overall = float(max(0.0, min(100.0, overall)))

    # Interpretative tier
    if overall >= 85:
        tier = 'Excellent'
    elif overall >= 70:
        tier = 'Good'
    elif overall >= 50:
        tier = 'Fair'
    else:
        tier = 'Needs Improvement'

    print("\n=== ATHLETE PERFORMANCE SUMMARY ===")
    print(f"Performance score: {overall:.1f}/100  ({tier})")
    print(f"Components -> Accuracy: {accuracy:.1f} | Form: {avg_quality:.1f} | Symmetry: {symmetry:.1f} | Consistency: {consistency:.1f} | Endurance: {endurance:.1f}")
    print(f"Average depth: {avg_depth:.1f}° | Fatigue score: {fatigue}/10")

    # Actionable recommendations
    tips = []
    if avg_quality < 75:
        tips.append('Work on technique: slow down and focus on depth')
    if symmetry < 80:
        tips.append('Balance training: address left-right imbalance')
    if consistency < 80:
        tips.append('Try to maintain a steady tempo across reps')
    if fatigue >= 6:
        tips.append('Endurance training: include rest or lighter sets')
    if accuracy < 80:
        tips.append('Aim for full range of motion on each rep')

    if tips:
        print('\nRecommended focus areas:')
        for t in tips:
            print(f" - {t}")
    else:
        print('\nNo major issues detected. Keep it up!')

    print('===================================\n')



def draw_advanced_metrics(frame, analyzer, terminal=False):
    """Draw all metrics on frame or print to terminal when `terminal=True`.

    If `terminal=True`, metrics are printed to stdout and the frame is returned
    unchanged (no overlay). This keeps the GUI clean while still exposing
    detailed results in the console.
    """
    if terminal:
        print_advanced_metrics(analyzer)
        return frame

    stats = analyzer.get_statistics()
    h, w = frame.shape[:2]

    # Main counter (large)
    cv2.putText(frame, str(analyzer.counter), (w - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 3)
    cv2.putText(frame, "Reps", (w - 150, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Correct vs Incorrect
    if stats:
        correct = stats.get('correct_reps', 0)
        incorrect = stats.get('incorrect_reps', 0)
        cv2.putText(frame, f"Good: {correct}  Bad: {incorrect}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Average metrics
        cv2.putText(frame, f"Avg Time: {stats.get('avg_rep_time', 0):.2f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Form: {stats.get('avg_form_quality', 0):.0f}/100", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Symmetry: {stats.get('avg_symmetry', 0):.0f}/100", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Fatigue indicator
        fatigue = stats.get('fatigue_score', 0)
        fatigue_color = (0, 255, 0) if fatigue < 4 else (0, 165, 255) if fatigue < 7 else (0, 0, 255)
        cv2.putText(frame, f"Fatigue: {int(fatigue)}/10", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)

    return frame


def draw_real_time_feedback(frame, feedback_text):
    """Draw real-time feedback prominently"""
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Feedback text
    cv2.putText(frame, feedback_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame
