
import cv2
import numpy as np
import time
from ultralytics import YOLO

class PlankTracker:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.reset_stats()

    def reset_stats(self):
        self.total_hold_time = 0
        self.is_holding = False
        self.holds = []  # list of dicts: start, end, duration, angles
        self.current_hold_start = None
        self.current_hold_angles = []
        self.session_time = 0.0

    def get_midpoint(self, kpts, idx1, idx2):
        """
        Calculates the center point between two keypoints (e.g., Left Hip & Right Hip).
        If one is missing, it uses the other.
        """
        p1 = kpts[idx1] # [x, y, conf]
        p2 = kpts[idx2]
        
        # Check visibility (coordinates must be > 0)
        p1_vis = p1[0] > 0
        p2_vis = p2[0] > 0
        
        if p1_vis and p2_vis:
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        elif p1_vis:
            return (p1[0], p1[1])
        elif p2_vis:
            return (p2[0], p2[1])
        else:
            return None

    def calculate_angle(self, a, b, c):
        """ Calculates angle at point b (Mid-Hip) """
        if not a or not b or not c: return 0
        
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def process_source(self, source_path=0, is_webcam=True):
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        frame_duration = 1.0 / fps
        
        window_name = "Plank Tracker (Center-Body Mode)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n--- PROCESSING (Center Mode) ---")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if is_webcam: frame = cv2.flip(frame, 1)

            results = self.model(frame, verbose=False)
            overlay = frame.copy()
            
            status = "Get Ready"
            color = (200, 200, 200)
            angle = 0

            if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                
                # --- GET MIDPOINTS (Center of Body) ---
                # Shoulders: Left(5), Right(6)
                # Elbows:    Left(7), Right(8)
                # Hips:      Left(11), Right(12)
                # Knees:     Left(13), Right(14)
                
                sho = self.get_midpoint(kpts, 5, 6)
                elb = self.get_midpoint(kpts, 7, 8)
                hip = self.get_midpoint(kpts, 11, 12)
                knee = self.get_midpoint(kpts, 13, 14)
                
                # Draw the "Spine" (The center line we are tracking)
                if sho and hip and knee:
                    # Draw Line
                    pt_sho = (int(sho[0]), int(sho[1]))
                    pt_hip = (int(hip[0]), int(hip[1]))
                    pt_knee = (int(knee[0]), int(knee[1]))
                    
                    cv2.line(overlay, pt_sho, pt_hip, (255, 255, 255), 2)
                    cv2.line(overlay, pt_hip, pt_knee, (255, 255, 255), 2)
                    
                    # Draw Center Points
                    cv2.circle(overlay, pt_sho, 5, (0, 255, 255), -1)
                    cv2.circle(overlay, pt_hip, 8, (0, 0, 255), -1) # The red dot in middle
                    cv2.circle(overlay, pt_knee, 5, (0, 255, 255), -1)

                    # --- CALCULATIONS ---
                    angle = self.calculate_angle(sho, hip, knee)
                    
                    # Elevation Logic (Center Shoulder vs Center Knee)
                    # Valid Plank: Shoulders are higher (lower Y) than Knees
                    # Valid Plank: Shoulders are higher than Elbows (Arms are vertical-ish)
                    
                    is_elevated = False
                    if elb:
                         # Shoulder above elbow check
                        arm_check = (elb[1] - sho[1]) > 20
                        # Knee vs Shoulder slope check
                        slope_check = (knee[1] - sho[1]) > 20
                        is_elevated = arm_check and slope_check
                    
                    # --- STATUS LOGIC ---
                    # Update session time
                    self.session_time += frame_duration

                    # Determine holding: use angle and elevation checks
                    is_angle_ok = (160 < angle < 190)
                    if is_angle_ok and is_elevated:
                        status = "PERFECT FORM"
                        color = (0, 255, 0)
                        # enter hold
                        if not self.is_holding:
                            self.is_holding = True
                            # start hold (relative to session_time)
                            self.current_hold_start = self.session_time
                            self.current_hold_angles = [angle]
                        else:
                            self.current_hold_angles.append(angle)
                        # increment total hold time
                        self.total_hold_time += frame_duration
                    else:
                        # leaving hold if we were holding
                        if self.is_holding:
                            self.is_holding = False
                            end_t = self.session_time
                            start_t = self.current_hold_start or (end_t - frame_duration)
                            duration = max(0.0, end_t - start_t)
                            avg_angle = float(np.mean(self.current_hold_angles)) if self.current_hold_angles else float(angle)
                            self.holds.append({
                                'start': start_t,
                                'end': end_t,
                                'duration': duration,
                                'avg_angle': avg_angle,
                            })
                            self.current_hold_start = None
                            self.current_hold_angles = []

                        # non-holding status reasons
                        if angle <= 160:
                            status = "HIPS TOO HIGH"
                            color = (0, 165, 255)
                        elif angle >= 190:
                            status = "SAGGING"
                            color = (0, 0, 255)
                        else:
                            status = "NOT HOLDING"
                            color = (100, 100, 100)

                    # Text Angle
                    cv2.putText(overlay, f"{int(angle)} deg", (pt_hip[0], pt_hip[1]-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Dashboard
            cv2.rectangle(overlay, (0, 0), (450, 100), (0,0,0), -1)
            cv2.putText(overlay, f"Time: {self.total_hold_time:.1f}s", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(overlay, status, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(window_name, overlay)
            delay = 1 if is_webcam else int(1000/fps)
            if cv2.waitKey(delay) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()
        # If currently holding when video ends, close that hold
        if self.is_holding and self.current_hold_start is not None:
            end_t = self.session_time
            start_t = self.current_hold_start
            duration = max(0.0, end_t - start_t)
            avg_angle = float(np.mean(self.current_hold_angles)) if self.current_hold_angles else 0.0
            self.holds.append({
                'start': start_t,
                'end': end_t,
                'duration': duration,
                'avg_angle': avg_angle,
            })
            self.is_holding = False

        self.print_statistics()

    def print_statistics(self):
        import statistics, json, os

        print(f"\nTotal Hold Time: {self.total_hold_time:.2f} seconds")
        total_holds = len(self.holds)
        durations = [h['duration'] for h in self.holds]
        avg_hold = float(np.mean(durations)) if durations else 0.0
        avg_angle = float(np.mean([h['avg_angle'] for h in self.holds])) if self.holds else 0.0

        consistency = 100.0
        if len(durations) > 1 and statistics.mean(durations) > 0:
            dur_std = float(np.std(durations))
            consistency = max(0.0, 100.0 - (dur_std / float(np.mean(durations))) * 100.0)

        percent_good = 0.0
        if self.holds:
            percent_good = sum(1 for h in self.holds if (h['avg_angle'] / 180.0) * 100.0 >= 75) / len(self.holds) * 100.0

        longest = max(durations) if durations else 0.0
        median = float(statistics.median(durations)) if durations else 0.0

        # Overall weighted score
        target_hold = 60.0
        hold_score = max(0.0, min(100.0, (avg_hold / target_hold) * 100.0))
        overall = (avg_angle * 0.40 + hold_score * 0.30 + consistency * 0.15 + percent_good * 0.15)
        overall = float(max(0.0, min(100.0, overall)))

        print(f"Total holds: {total_holds}")
        print(f"Avg hold: {avg_hold:.2f}s | Longest: {longest:.2f}s | Median: {median:.2f}s")
        print(f"Avg angle: {avg_angle:.1f}° | Consistency: {consistency:.1f} | %Good holds: {percent_good:.1f}")
        print(f"Overall score: {overall:.1f}/100")

        # Export JSON report next to script
        try:
            report = {
                'total_hold_time': self.total_hold_time,
                'total_holds': total_holds,
                'avg_hold': avg_hold,
                'avg_angle': avg_angle,
                'consistency': consistency,
                'percent_good': percent_good,
                'overall_score': overall,
                'holds': self.holds,
            }
            out_path = os.path.join(os.path.dirname(__file__), f'planck_report_{int(time.time())}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f'Report exported to: {out_path}')
        except Exception as e:
            print('Failed to export report:', e)

def main():
    tracker = PlankTracker()
    while True:
        c = input("\nW: Webcam | V: Video | Q: Quit > ").lower()
        if c == 'w': tracker.process_source(0, True)
        elif c == 'v': 
            f = input("Filename: ")
            try: tracker.process_source(f, False)
            except: print("File error.")
        elif c == 'q': break

if __name__ == "__main__":
    main()
