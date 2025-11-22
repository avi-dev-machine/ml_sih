import cv2
import numpy as np
import time
from ultralytics import YOLO

class SquatTracker:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.reset_stats()

    def reset_stats(self):
        self.total_squats = 0
        self.is_squatting = False
        self.squat_started = False
        self.squats_data = []  # stores depth, angles, form quality
        self.current_squat_start = None
        self.session_time = 0.0
        self.start_time = None
        self.lowest_knee_angle = None
        self.form_issues = []

    def get_midpoint(self, kpts, idx1, idx2):
        """Calculate midpoint between two keypoints."""
        p1 = kpts[idx1]
        p2 = kpts[idx2]
        
        p1_vis = p1[0] > 0 and p1[1] > 0
        p2_vis = p2[0] > 0 and p2[1] > 0
        
        if p1_vis and p2_vis:
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        elif p1_vis:
            return (p1[0], p1[1])
        elif p2_vis:
            return (p2[0], p2[1])
        else:
            return None

    def calculate_angle(self, a, b, c):
        """Calculate angle at point b."""
        if not a or not b or not c: 
            return 0
        
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: 
            angle = 360 - angle
        return angle

    def draw_skeleton(self, overlay, kpts):
        """Draw the full human skeleton with key points highlighted."""
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]

        # Draw connections
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(kpts) and end_idx < len(kpts):
                start_pt = kpts[start_idx]
                end_pt = kpts[end_idx]
                
                if start_pt[0] > 0 and start_pt[1] > 0 and end_pt[0] > 0 and end_pt[1] > 0:
                    # Highlight leg connections
                    color = (0, 255, 255) if start_idx in [11, 12, 13, 14, 15, 16] else (100, 200, 100)
                    thickness = 3 if start_idx in [11, 12, 13, 14, 15, 16] else 2
                    cv2.line(overlay, 
                            (int(start_pt[0]), int(start_pt[1])),
                            (int(end_pt[0]), int(end_pt[1])),
                            color, thickness)

        # Draw keypoints
        for i, kpt in enumerate(kpts):
            if kpt[0] > 0 and kpt[1] > 0:
                # Highlight hips and knees
                if i in [11, 12]:  # Hips
                    color = (255, 0, 255)
                    radius = 7
                elif i in [13, 14]:  # Knees
                    color = (0, 0, 255)
                    radius = 7
                elif i in [15, 16]:  # Ankles
                    color = (255, 255, 0)
                    radius = 6
                else:
                    color = (0, 255, 0)
                    radius = 4
                cv2.circle(overlay, (int(kpt[0]), int(kpt[1])), radius, color, -1)

    def analyze_squat_form(self, hip_mid, knee_mid, ankle_mid, hip_angle, knee_angle):
        """Analyze squat form and return feedback."""
        issues = []
        form_score = 100.0
        
        # Check knee angle (should be between 70-110 degrees for good depth)
        if knee_angle > 110:
            issues.append("Not deep enough")
            form_score -= 20
        elif knee_angle < 70:
            issues.append("Too deep - risk of injury")
            form_score -= 10
        
        # Check if knees are forward of ankles (knee should not go too far forward)
        if knee_mid and ankle_mid:
            knee_forward = knee_mid[0] - ankle_mid[0]
            if abs(knee_forward) > 100:  # Knees too far forward
                issues.append("Knees too far forward")
                form_score -= 15
        
        # Check hip angle (torso lean)
        if hip_angle < 140:
            issues.append("Leaning too far forward")
            form_score -= 10
        elif hip_angle > 190:
            issues.append("Stand more upright")
            form_score -= 5
        
        # Check hip vs knee depth (hips should go below knees for full squat)
        if hip_mid and knee_mid:
            depth_ratio = (hip_mid[1] - knee_mid[1])
            if depth_ratio < -20:  # Hip below knee = good depth
                depth_quality = "Full depth"
            elif depth_ratio < 20:
                depth_quality = "Parallel"
            else:
                depth_quality = "Shallow"
                form_score -= 15
        else:
            depth_quality = "Unknown"
        
        return issues, form_score, depth_quality

    def process_source(self, source_path=0, is_webcam=True):
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        frame_duration = 1.0 / fps
        
        window_name = "Squat Counter & Form Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n{'='*50}")
        print(f"SQUAT COUNTER & FORM TRACKER")
        print(f"{'='*50}")
        print("Stand in view and start squatting!")
        print("Press 'R' to reset counter | 'Q' to quit")
        
        self.start_time = time.time()
        standing_position = True  # Track if we're in standing position

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if is_webcam: frame = cv2.flip(frame, 1)

            results = self.model(frame, verbose=False)
            overlay = frame.copy()
            
            status = "Stand Ready"
            color = (200, 200, 200)
            h, w = frame.shape[:2]

            if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                
                # Draw skeleton
                self.draw_skeleton(overlay, kpts)
                
                # Get key points (using midpoints for center body tracking)
                # Hip: 11 (left), 12 (right)
                # Knee: 13 (left), 14 (right)
                # Ankle: 15 (left), 16 (right)
                # Shoulder: 5 (left), 6 (right)
                
                hip_mid = self.get_midpoint(kpts, 11, 12)
                knee_mid = self.get_midpoint(kpts, 13, 14)
                ankle_mid = self.get_midpoint(kpts, 15, 16)
                shoulder_mid = self.get_midpoint(kpts, 5, 6)
                
                if hip_mid and knee_mid and ankle_mid and shoulder_mid:
                    self.session_time += frame_duration
                    
                    # Calculate angles
                    # Knee angle: ankle -> knee -> hip
                    knee_angle = self.calculate_angle(ankle_mid, knee_mid, hip_mid)
                    
                    # Hip angle: knee -> hip -> shoulder (torso angle)
                    hip_angle = self.calculate_angle(knee_mid, hip_mid, shoulder_mid)
                    
                    # Draw angle lines and circles
                    pt_hip = (int(hip_mid[0]), int(hip_mid[1]))
                    pt_knee = (int(knee_mid[0]), int(knee_mid[1]))
                    pt_ankle = (int(ankle_mid[0]), int(ankle_mid[1]))
                    pt_shoulder = (int(shoulder_mid[0]), int(shoulder_mid[1]))
                    
                    # Draw angle visualization
                    cv2.line(overlay, pt_hip, pt_shoulder, (255, 255, 255), 2)
                    cv2.line(overlay, pt_hip, pt_knee, (255, 255, 255), 2)
                    cv2.line(overlay, pt_knee, pt_ankle, (255, 255, 255), 2)
                    
                    cv2.circle(overlay, pt_hip, 8, (255, 0, 255), -1)
                    cv2.circle(overlay, pt_knee, 8, (0, 0, 255), -1)
                    cv2.circle(overlay, pt_ankle, 8, (255, 255, 0), -1)
                    
                    # Squat detection logic
                    # Standing: knee angle > 160 degrees
                    # Squatting: knee angle < 120 degrees
                    
                    is_standing = knee_angle > 160
                    is_in_squat = knee_angle < 120
                    
                    # State machine for squat counting
                    if is_in_squat and standing_position:
                        # Started going down
                        standing_position = False
                        self.squat_started = True
                        self.current_squat_start = self.session_time
                        self.lowest_knee_angle = knee_angle
                        self.form_issues = []
                        status = "GOING DOWN"
                        color = (0, 165, 255)
                        
                    elif self.squat_started and not standing_position:
                        # In the squat, track lowest angle
                        if knee_angle < self.lowest_knee_angle:
                            self.lowest_knee_angle = knee_angle
                        status = "IN SQUAT"
                        color = (0, 255, 255)
                        
                    elif is_standing and not standing_position:
                        # Completed the squat - coming back up
                        standing_position = True
                        
                        if self.squat_started:
                            self.squat_started = False
                            self.total_squats += 1
                            
                            # Analyze form
                            issues, form_score, depth_quality = self.analyze_squat_form(
                                hip_mid, knee_mid, ankle_mid, hip_angle, self.lowest_knee_angle
                            )
                            
                            squat_duration = self.session_time - self.current_squat_start if self.current_squat_start else 0
                            
                            self.squats_data.append({
                                'number': self.total_squats,
                                'time': self.session_time,
                                'duration': squat_duration,
                                'lowest_knee_angle': self.lowest_knee_angle,
                                'form_score': form_score,
                                'depth_quality': depth_quality,
                                'issues': issues
                            })
                            
                            print(f"✓ Squat #{self.total_squats} | Depth: {self.lowest_knee_angle:.1f}° | "
                                  f"Form: {form_score:.0f}/100 | {depth_quality}")
                            if issues:
                                print(f"  ⚠ Issues: {', '.join(issues)}")
                            
                            self.form_issues = issues
                        
                        status = "STANDING"
                        color = (0, 255, 0)
                    
                    elif is_standing:
                        status = "READY"
                        color = (255, 255, 255)
                    
                    # Display angles
                    cv2.putText(overlay, f"Knee: {int(knee_angle)}°", 
                               (pt_knee[0] + 15, pt_knee[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.putText(overlay, f"Hip: {int(hip_angle)}°", 
                               (pt_hip[0] + 15, pt_hip[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    # Show form feedback
                    if self.form_issues and len(self.form_issues) > 0:
                        y_offset = 200
                        for issue in self.form_issues[:3]:  # Show max 3 issues
                            cv2.putText(overlay, f"⚠ {issue}", (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            y_offset += 30

            # Dashboard
            cv2.rectangle(overlay, (0, 0), (500, 170), (0, 0, 0), -1)
            cv2.putText(overlay, f"Squats: {self.total_squats}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
            
            # Calculate average form score
            if self.squats_data:
                avg_form = np.mean([s['form_score'] for s in self.squats_data])
                form_color = (0, 255, 0) if avg_form >= 80 else (0, 165, 255) if avg_form >= 60 else (0, 0, 255)
                cv2.putText(overlay, f"Avg Form: {avg_form:.0f}/100", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
            
            # Display last squat depth if available
            if self.squats_data:
                last_depth = self.squats_data[-1]['depth_quality']
                cv2.putText(overlay, f"Last: {last_depth}", (20, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(overlay, status, (20, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(window_name, overlay)
            delay = 1 if is_webcam else int(1000/fps)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('r'):  # Reset counter
                self.reset_stats()
                standing_position = True
                print("\n--- Counter Reset ---")
        
        cap.release()
        cv2.destroyAllWindows()

        self.print_statistics()

    def print_statistics(self):
        import json, os

        print(f"\n{'='*60}")
        print(f"SQUAT SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Squats: {self.total_squats}")
        
        if self.squats_data:
            # Calculate statistics
            depths = [s['lowest_knee_angle'] for s in self.squats_data]
            form_scores = [s['form_score'] for s in self.squats_data]
            
            avg_depth = float(np.mean(depths))
            avg_form = float(np.mean(form_scores))
            
            # Count depth quality
            depth_counts = {}
            for s in self.squats_data:
                quality = s['depth_quality']
                depth_counts[quality] = depth_counts.get(quality, 0) + 1
            
            print(f"\nDepth Analysis:")
            print(f"  Average Knee Angle: {avg_depth:.1f}°")
            for quality, count in depth_counts.items():
                print(f"  {quality}: {count} ({count/len(self.squats_data)*100:.1f}%)")
            
            print(f"\nForm Analysis:")
            print(f"  Average Form Score: {avg_form:.1f}/100")
            
            # Most common issues
            all_issues = []
            for s in self.squats_data:
                all_issues.extend(s['issues'])
            
            if all_issues:
                from collections import Counter
                issue_counts = Counter(all_issues)
                print(f"\nMost Common Issues:")
                for issue, count in issue_counts.most_common(3):
                    print(f"  • {issue}: {count} times")
            else:
                print(f"  ✓ No major form issues detected!")
            
            # Export report
            try:
                report = {
                    'total_squats': self.total_squats,
                    'avg_depth_angle': avg_depth,
                    'avg_form_score': avg_form,
                    'depth_distribution': depth_counts,
                    'squats': self.squats_data
                }
                out_path = os.path.join(os.path.dirname(__file__), f'squat_report_{int(time.time())}.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
                print(f'\n✓ Report exported to: {out_path}')
            except Exception as e:
                print(f'✗ Failed to export report: {e}')
        else:
            print("No squats recorded.")
        
        print(f"{'='*60}\n")

def main():
    tracker = SquatTracker()
    while True:
        c = input("\nW: Webcam | V: Video | Q: Quit > ").lower()
        if c == 'w': 
            tracker.reset_stats()
            tracker.process_source(0, True)
        elif c == 'v': 
            f = input("Filename: ")
            try: 
                tracker.reset_stats()
                tracker.process_source(f, False)
            except Exception as e: 
                print(f"File error: {e}")
        elif c == 'q': 
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()