import cv2
import numpy as np
import time
import os
import json
import statistics
from ultralytics import YOLO

# --- Jumping Jack Logic Constants (Based on YOLOv8-pose Keypoints) ---
# Keypoints:
# Shoulders: Left(5), Right(6)
# Wrists:    Left(9), Right(10)
# Hips:      Left(11), Right(12)
# Ankles:    Left(15), Right(16)

class JumpingJackTracker:
    def __init__(self):
        # Using the same pose estimation model
        self.model = YOLO("yolov8n-pose.pt")
        self.reset_stats()

    def reset_stats(self):
        self.total_jumps = 0
        self.jumps = []  # list of dicts: start_time, end_time, duration
        self.is_high_pose = False
        self.is_low_pose = True # Start assuming low pose
        self.start_time = time.time()
        self.session_time = 0.0
        self.last_jump_time = 0.0

    def get_midpoint(self, kpts, idx1, idx2):
        """
        Calculates the center point between two keypoints.
        """
        p1 = kpts[idx1] # [x, y, conf]
        p2 = kpts[idx2]
        
        # Check visibility (coordinates must be > 0 and confidence > 0.5)
        p1_vis = p1[0] > 0 and p1[2] > 0.5
        p2_vis = p2[0] > 0 and p2[2] > 0.5
        
        if p1_vis and p2_vis:
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        elif p1_vis:
            return (p1[0], p1[1])
        elif p2_vis:
            return (p2[0], p2[1])
        else:
            return None

    def check_arm_high(self, kpts):
        """Checks if both hands (wrists 9, 10) are above the shoulders (5, 6)."""
        L_wrist, R_wrist = kpts[9], kpts[10]
        L_shoulder, R_shoulder = kpts[5], kpts[6]
        
        # Check if wrists are present and above shoulders (lower Y is higher)
        arm_high = (L_wrist[2] > 0.5 and R_wrist[2] > 0.5 and
                    L_shoulder[2] > 0.5 and R_shoulder[2] > 0.5)

        if arm_high:
            # Check Y-coordinate (lower Y means higher position on the screen)
            L_arm_check = L_wrist[1] < L_shoulder[1]
            R_arm_check = R_wrist[1] < R_shoulder[1]
            # Requires both arms to be above the respective shoulder
            return L_arm_check and R_arm_check
        return False

    def check_leg_wide(self, kpts):
        """Checks if the ankles (15, 16) are significantly wider than the hips (11, 12)."""
        L_ankle, R_ankle = kpts[15], kpts[16]
        L_hip, R_hip = kpts[11], kpts[12]

        # Check visibility
        leg_wide = (L_ankle[2] > 0.5 and R_ankle[2] > 0.5 and
                    L_hip[2] > 0.5 and R_hip[2] > 0.5)
        
        if leg_wide:
            # Calculate the horizontal distance between the ankles and hips
            ankle_distance = abs(R_ankle[0] - L_ankle[0])
            hip_distance = abs(R_hip[0] - L_hip[0])
            
            # Legs are wide if the ankle distance is significantly greater than hip distance
            # A factor of 1.5-2.0 is usually a good heuristic.
            return ankle_distance > (hip_distance * 1.5)
        return False

    def process_source(self, source_path=0, is_webcam=True):
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        frame_duration = 1.0 / fps
        
        window_name = "Jumping Jack Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n--- PROCESSING JUMPING JACKS ---")
        
        self.start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if is_webcam: frame = cv2.flip(frame, 1)

            results = self.model(frame, verbose=False)
            overlay = frame.copy()
            
            status = "Get Ready"
            color = (200, 200, 200)

            if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy() # [x, y, conf]
                
                # --- JUMPING JACK LOGIC ---
                self.session_time = time.time() - self.start_time
                
                arms_up = self.check_arm_high(kpts)
                legs_out = self.check_leg_wide(kpts)

                is_jump_pose = arms_up and legs_out
                is_reset_pose = (not arms_up) and (not legs_out)

                # State Machine for Counting Jumps:
                # 1. Detect **High Pose** (Arms Up & Legs Wide)
                # 2. Transition to **Low Pose** (Arms Down & Legs Together) -> **Counts as one full jump**

                if is_jump_pose:
                    status = "HIGH POSE (Arms Up, Legs Out)"
                    color = (0, 255, 255) # Yellow
                    self.is_high_pose = True
                
                elif is_reset_pose:
                    status = "LOW POSE (Resting)"
                    color = (0, 255, 0) # Green
                    
                    # If we were in a High Pose and now we are in a Reset/Low Pose, a jump is complete.
                    if self.is_high_pose:
                        self.total_jumps += 1
                        self.is_high_pose = False # Reset high pose state
                        current_time = self.session_time
                        
                        # Store jump info
                        jump_duration = current_time - self.last_jump_time
                        self.jumps.append({'end_time': current_time, 'duration': jump_duration})
                        self.last_jump_time = current_time
                        
                        print(f"Jump: {self.total_jumps} | Duration: {jump_duration:.2f}s")
                    
                else:
                    status = "TRANSITION / BAD FORM"
                    color = (0, 0, 255) # Red
                
                # Draw Keypoints for visualization
                # Note: The YOLO model's plot function is usually better, but we'll manually draw the joints of interest
                
                # Visualize points that are part of the state check
                points_to_draw = [5, 6, 9, 10, 11, 12, 15, 16] 
                for idx in points_to_draw:
                    pt = kpts[idx]
                    if pt[2] > 0.5: # Confidence check
                        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

            # Dashboard
            cv2.rectangle(overlay, (0, 0), (450, 100), (0,0,0), -1)
            cv2.putText(overlay, f"Jumps: {self.total_jumps}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(overlay, status, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(window_name, overlay)
            delay = 1 if is_webcam else int(1000/fps)
            if cv2.waitKey(delay) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()
        self.print_statistics()
        
        # 

    def print_statistics(self):
        print(f"\n--- SESSION STATISTICS ---")
        
        # Calculate total session time for metric context
        session_duration = self.session_time
        if session_duration > 0 and self.total_jumps > 0:
            # Exclude the first "jump" duration as it's from 0.0 to the first rep
            durations = [j['duration'] for j in self.jumps if j['end_time'] > self.jumps[0]['end_time']] 
        else:
            durations = []
            
        total_jumps = self.total_jumps
        avg_rate = total_jumps / session_duration * 60 if session_duration > 0 else 0.0 # Jumps per minute
        
        avg_duration = float(np.mean(durations)) if durations else 0.0
        median_duration = float(statistics.median(durations)) if durations else 0.0
        
        consistency = 100.0
        if len(durations) > 1 and avg_duration > 0:
            dur_std = float(np.std(durations))
            # Lower standard deviation means higher consistency
            consistency = max(0.0, 100.0 - (dur_std / avg_duration) * 100.0)

        print(f"Total Jumps: {total_jumps}")
        print(f"Total Session Time: {session_duration:.2f}s")
        print(f"Average Rate: {avg_rate:.1f} jumps/min")
        print(f"Average Jump Duration: {avg_duration:.2f}s | Median: {median_duration:.2f}s")
        print(f"Consistency Score: {consistency:.1f}/100")
        
        # Export JSON report
        try:
            report = {
                'total_jumps': total_jumps,
                'session_duration': session_duration,
                'avg_rate_jumps_per_min': avg_rate,
                'avg_jump_duration_sec': avg_duration,
                'consistency_score': consistency,
                'jumps': self.jumps,
            }
            # Use current file directory for output
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'jjack_report_{int(time.time())}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f'Report exported to: {out_path}')
        except Exception as e:
            print('Failed to export report:', e)

def main():
    tracker = JumpingJackTracker()
    while True:
        c = input("\nJUMPING JACK TRACKER | W: Webcam | V: Video | Q: Quit > ").lower()
        if c == 'w': 
            tracker.reset_stats()
            tracker.process_source(0, True)
        elif c == 'v': 
            f = input("Filename: ")
            try: 
                tracker.reset_stats()
                tracker.process_source(f, False)
            except: 
                print("File error or model loading failed.")
        elif c == 'q': 
            break

if __name__ == "__main__":
    main()