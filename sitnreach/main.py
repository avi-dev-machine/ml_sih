import cv2
import numpy as np
import time
from ultralytics import YOLO

class SitReachTracker:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.reset_stats()

    def reset_stats(self):
        self.max_reach = 0.0
        self.baseline_reach = None
        self.reaches = []  # list of dicts: timestamp, reach, knee_angle, form_score
        self.current_trial = []
        self.trial_maxes = []  # max reach per trial
        self.session_time = 0.0
        self.frame_count = 0

    def get_midpoint(self, kpts, idx1, idx2):
        """Calculate midpoint between two keypoints"""
        p1 = kpts[idx1]
        p2 = kpts[idx2]
        
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

    def calculate_angle(self, a, b, c):
        """Calculate angle at point b"""
        if not a or not b or not c: return 0
        
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance"""
        if not p1 or not p2: return 0
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def process_source(self, source_path=0, is_webcam=True):
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        frame_duration = 1.0 / fps
        
        window_name = "Sit-and-Reach Test Analyzer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n--- PROCESSING SIT-AND-REACH TEST ---")
        print("Press 'T' to mark trial | 'Q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if is_webcam: frame = cv2.flip(frame, 1)

            results = self.model(frame, verbose=False)
            overlay = frame.copy()
            
            status = "Position Yourself"
            color = (200, 200, 200)
            reach_dist = 0.0
            knee_angle = 0.0
            form_score = 0.0

            if results[0].keypoints is not None:
                kp = results[0].keypoints
                # Prefer full (x,y,conf) keypoint array when available.
                # Some ultralytics builds expose x,y only in `kp.xy` and confidences in `kp.conf`.
                kpts = None
                try:
                    if hasattr(kp, 'xy') and kp.xy.numel() > 0:
                        kpts = kp.xy[0].cpu().numpy()
                except Exception:
                    kpts = None

                # If xy exists but only contains x,y (shape[* ,2]), try to append confidences
                if kpts is not None and kpts.shape[1] == 2:
                    if hasattr(kp, 'conf'):
                        try:
                            confs = kp.conf[0].cpu().numpy()
                            # ensure confs is shape (num_kpts,)
                            confs = confs.reshape((-1, 1))
                            kpts = np.concatenate([kpts, confs], axis=1)
                        except Exception:
                            # fallback: add ones as confidences
                            kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
                    else:
                        kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)

                # If the model already returned x,y,conf together (shape[* ,3]) use it
                if kpts is None:
                    try:
                        # some versions expose all keypoints in `kp.data`
                        kpts = kp.data[0].cpu().numpy()
                    except Exception:
                        kpts = None

                # If we still don't have a usable kpts array, skip this frame
                if kpts is None or kpts.size == 0:
                    kpts = None
                
                # Keypoint indices (COCO format)
                # Shoulders: L(5), R(6) | Hips: L(11), R(12) | Knees: L(13), R(14)
                # Ankles: L(15), R(16) | Wrists: L(9), R(10)
                
                sho = self.get_midpoint(kpts, 5, 6)
                hip = self.get_midpoint(kpts, 11, 12)
                knee = self.get_midpoint(kpts, 13, 14)
                ankle = self.get_midpoint(kpts, 15, 16)
                wrist = self.get_midpoint(kpts, 9, 10)
                
                # Get individual points for detailed analysis
                # Defensive indexing: ensure keypoint index exists and confidence column present
                def kp_with_conf(arr, idx, conf_thresh=0.5):
                    if arr is None: return None
                    if idx < 0 or idx >= arr.shape[0]: return None
                    if arr.shape[1] < 3:
                        # no confidence available, accept the point
                        return (arr[idx][0], arr[idx][1])
                    return (arr[idx][0], arr[idx][1]) if arr[idx][2] > conf_thresh else None

                left_hip = kp_with_conf(kpts, 11)
                left_knee = kp_with_conf(kpts, 13)
                left_ankle = kp_with_conf(kpts, 15)
                
                # Draw skeleton
                if sho and hip and knee and ankle:
                    pt_sho = (int(sho[0]), int(sho[1]))
                    pt_hip = (int(hip[0]), int(hip[1]))
                    pt_knee = (int(knee[0]), int(knee[1]))
                    pt_ankle = (int(ankle[0]), int(ankle[1]))
                    
                    # Draw spine line (shoulder to hip)
                    cv2.line(overlay, pt_sho, pt_hip, (255, 255, 255), 2)
                    # Draw leg line (hip to knee to ankle)
                    cv2.line(overlay, pt_hip, pt_knee, (255, 255, 255), 2)
                    cv2.line(overlay, pt_knee, pt_ankle, (255, 255, 255), 2)
                    
                    # Draw keypoints
                    cv2.circle(overlay, pt_sho, 5, (0, 255, 255), -1)
                    cv2.circle(overlay, pt_hip, 8, (0, 0, 255), -1)
                    cv2.circle(overlay, pt_knee, 5, (0, 255, 255), -1)
                    cv2.circle(overlay, pt_ankle, 5, (0, 255, 255), -1)
                    
                    if wrist:
                        pt_wrist = (int(wrist[0]), int(wrist[1]))
                        # Draw reach line (hip to wrist)
                        cv2.line(overlay, pt_hip, pt_wrist, (0, 255, 0), 3)
                        cv2.circle(overlay, pt_wrist, 8, (255, 0, 255), -1)

                    # --- METRIC CALCULATIONS ---
                    self.session_time += frame_duration
                    self.frame_count += 1
                    
                    # 1. REACH DISTANCE (horizontal distance from hip to wrist)
                    if wrist and hip:
                        reach_dist = abs(wrist[0] - hip[0])
                        
                        # Set baseline from first 30 frames
                        if self.frame_count == 30 and self.baseline_reach is None:
                            self.baseline_reach = reach_dist
                        
                        if reach_dist > self.max_reach:
                            self.max_reach = reach_dist
                    
                    # 2. KNEE ANGLE (should be >= 175° for valid form)
                    if left_hip and left_knee and left_ankle:
                        lh = (left_hip[0], left_hip[1])
                        lk = (left_knee[0], left_knee[1])
                        la = (left_ankle[0], left_ankle[1])
                        knee_angle = self.calculate_angle(lh, lk, la)
                    
                    # 3. HIP FLEXION ANGLE (shoulder-hip-knee)
                    hip_angle = self.calculate_angle(sho, hip, knee) if sho and hip and knee else 0
                    
                    # 4. TOE ALIGNMENT CHECK (ankle width vs hip width)
                    left_ankle_pt = kp_with_conf(kpts, 15)
                    right_ankle_pt = kp_with_conf(kpts, 16)
                    left_hip_pt = kp_with_conf(kpts, 11)
                    right_hip_pt = kp_with_conf(kpts, 12)
                    
                    toe_alignment = 0.0
                    if left_ankle_pt is not None and right_ankle_pt is not None and left_hip_pt is not None and right_hip_pt is not None:
                        ankle_width = abs(right_ankle_pt[0] - left_ankle_pt[0])
                        hip_width = abs(right_hip_pt[0] - left_hip_pt[0])
                        if hip_width > 0:
                            toe_alignment = abs((ankle_width / hip_width - 1.0) * 90)  # deviation in degrees
                    
                    # 5. FORM VALIDITY SCORE (0-100)
                    form_score = 100.0
                    
                    # Penalize knee bend
                    if knee_angle > 0 and knee_angle < 175:
                        form_score -= (175 - knee_angle) * 2
                    
                    # Penalize toe misalignment (>20° rotation)
                    if toe_alignment > 20:
                        form_score -= (toe_alignment - 20) * 1.5
                    
                    # Bonus for good hip flexion (lower angle = more flexible)
                    if hip_angle < 90:
                        form_score += (90 - hip_angle) * 0.5
                    
                    form_score = max(0.0, min(100.0, form_score))
                    
                    # --- STATUS DETERMINATION ---
                    knee_valid = knee_angle >= 175 or knee_angle == 0
                    form_valid = form_score >= 70
                    
                    if knee_valid and form_valid:
                        status = "EXCELLENT FORM"
                        color = (0, 255, 0)
                    elif knee_valid and form_score >= 50:
                        status = "GOOD FORM"
                        color = (0, 255, 255)
                    elif not knee_valid:
                        status = "STRAIGHTEN KNEES"
                        color = (0, 0, 255)
                    else:
                        status = "ADJUST FORM"
                        color = (0, 165, 255)
                    
                    # Store reach data
                    if reach_dist > 0:
                        self.reaches.append({
                            'timestamp': self.session_time,
                            'reach': float(reach_dist),
                            'knee_angle': float(knee_angle),
                            'hip_angle': float(hip_angle),
                            'form_score': float(form_score)
                        })
                    
                    # Display angle on frame
                    if knee_angle > 0:
                        cv2.putText(overlay, f"Knee: {int(knee_angle)}deg", 
                                   (pt_knee[0]-80, pt_knee[1]-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Dashboard overlay
            dash_h = 150
            cv2.rectangle(overlay, (0, 0), (500, dash_h), (0, 0, 0), -1)
            
            y_pos = 35
            cv2.putText(overlay, f"Max Reach: {self.max_reach:.1f}px", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            y_pos += 35
            if reach_dist > 0:
                cv2.putText(overlay, f"Current: {reach_dist:.1f}px", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            y_pos += 35
            if form_score > 0:
                form_color = (0, 255, 0) if form_score >= 70 else (0, 165, 255)
                cv2.putText(overlay, f"Form: {form_score:.0f}%", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
            
            y_pos += 35
            cv2.putText(overlay, status, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Instructions
            cv2.putText(overlay, "T: Save Trial | Q: Quit", 
                       (10, overlay.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, overlay)
            
            delay = 1 if is_webcam else int(1000/fps)
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                # Save trial
                if self.max_reach > 0:
                    self.trial_maxes.append(self.max_reach)
                    print(f"\nTrial {len(self.trial_maxes)} saved: {self.max_reach:.1f}px")
                    self.max_reach = 0.0  # Reset for next trial
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.print_statistics()

    def print_statistics(self):
        import statistics, json, os

        print("\n" + "="*60)
        print("SIT-AND-REACH TEST PERFORMANCE REPORT")
        print("="*60)
        
        if not self.reaches:
            print("No data collected.")
            return
        
        # Extract metrics
        all_reaches = [r['reach'] for r in self.reaches]
        all_knee_angles = [r['knee_angle'] for r in self.reaches if r['knee_angle'] > 0]
        all_hip_angles = [r['hip_angle'] for r in self.reaches if r['hip_angle'] > 0]
        all_form_scores = [r['form_score'] for r in self.reaches]
        
        # 1. Maximum Reach Distance
        max_reach_overall = max(all_reaches) if all_reaches else 0.0
        print(f"\n1. Maximum Reach Distance: {max_reach_overall:.1f} pixels")
        
        # 2. Improvement Score
        if self.baseline_reach and max_reach_overall > 0:
            improvement = max_reach_overall - self.baseline_reach
            improvement_pct = (improvement / self.baseline_reach) * 100
            print(f"2. Improvement: {improvement:.1f}px ({improvement_pct:.1f}%)")
        
        # 3. Left-Right Symmetry (tracked in trials)
        if len(self.trial_maxes) > 1:
            trial_variance = float(np.std(self.trial_maxes))
            trial_mean = float(np.mean(self.trial_maxes))
            print(f"3. Trial Consistency: {trial_mean:.1f}px ± {trial_variance:.1f}px")
        
        # 4. Hold Stability (frame-to-frame variance in reach)
        if len(all_reaches) > 30:
            recent_reaches = all_reaches[-30:]
            stability_variance = float(np.std(recent_reaches))
            print(f"4. Hold Stability (variance): {stability_variance:.2f}px")
        
        # 5. Spine Curvature Quality (via form scores)
        avg_form_score = float(np.mean(all_form_scores)) if all_form_scores else 0.0
        print(f"5. Average Form Quality: {avg_form_score:.1f}%")
        
        # 6. Hip Flexion Angle
        avg_hip_angle = float(np.mean(all_hip_angles)) if all_hip_angles else 0.0
        print(f"6. Average Hip Flexion: {avg_hip_angle:.1f}°")
        
        # 7. Knee Bend Detection
        avg_knee_angle = float(np.mean(all_knee_angles)) if all_knee_angles else 0.0
        knee_valid = avg_knee_angle >= 175
        print(f"7. Average Knee Angle: {avg_knee_angle:.1f}° {'✓ Valid' if knee_valid else '✗ Bent'}")
        
        # 9. Smoothness of Movement
        if len(all_reaches) > 10:
            velocities = np.diff(all_reaches)
            smoothness = float(np.std(velocities))
            print(f"9. Movement Smoothness: {smoothness:.2f} (lower=smoother)")
        
        # 10. Trial Consistency
        if len(self.trial_maxes) > 1:
            trial_consistency = float(np.std(self.trial_maxes))
            print(f"10. Multi-Trial Variance: {trial_consistency:.2f}px")
        
        # 12. Overall Performance Score
        reach_score = min(100.0, (max_reach_overall / 300.0) * 100.0)  # normalize to 300px
        knee_score = 100.0 if knee_valid else 50.0
        overall_score = (reach_score * 0.40 + avg_form_score * 0.30 + 
                        knee_score * 0.20 + (100 - smoothness if len(all_reaches) > 10 else 80) * 0.10)
        overall_score = float(max(0.0, min(100.0, overall_score)))
        
        print(f"\n{'='*60}")
        print(f"OVERALL PERFORMANCE SCORE: {overall_score:.1f}/100")
        print(f"{'='*60}")
        
        # Trial breakdown
        if self.trial_maxes:
            print(f"\nTrials Completed: {len(self.trial_maxes)}")
            for i, trial_max in enumerate(self.trial_maxes, 1):
                print(f"  Trial {i}: {trial_max:.1f}px")
        
        # Export JSON report
        try:
            report = {
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'session_duration': self.session_time,
                'metrics': {
                    'max_reach_distance_px': float(max_reach_overall),
                    'baseline_reach_px': float(self.baseline_reach) if self.baseline_reach else None,
                    'improvement_px': float(improvement) if self.baseline_reach else None,
                    'avg_form_score': avg_form_score,
                    'avg_knee_angle_deg': avg_knee_angle,
                    'avg_hip_flexion_deg': avg_hip_angle,
                    'hold_stability_variance': float(stability_variance) if len(all_reaches) > 30 else None,
                    'movement_smoothness': float(smoothness) if len(all_reaches) > 10 else None,
                    'overall_score': overall_score
                },
                'trials': [{'trial': i+1, 'max_reach_px': float(t)} 
                          for i, t in enumerate(self.trial_maxes)],
                'detailed_reaches': self.reaches[-100:]  # Last 100 frames
            }
            
            out_path = os.path.join(os.path.dirname(__file__) or '.', 
                                   f'sit_reach_report_{int(time.time())}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f'\n✓ Report exported to: {out_path}')
        except Exception as e:
            print(f'✗ Failed to export report: {e}')

def main():
    tracker = SitReachTracker()
    print("\n" + "="*60)
    print("SIT-AND-REACH TEST ANALYZER")
    print("="*60)
    print("\nThis system tracks flexibility and form during sit-and-reach tests.")
    print("Position yourself sitting with legs extended, then reach forward.\n")
    
    while True:
        choice = input("\nW: Webcam | V: Video File | Q: Quit > ").lower()
        
        if choice == 'w':
            tracker.reset_stats()
            tracker.process_source(0, True)
        elif choice == 'v':
            video_path = input("Enter video filename: ").strip()
            try:
                tracker.reset_stats()
                tracker.process_source(video_path, False)
            except Exception as e:
                print(f"Error processing video: {e}")
        elif choice == 'q':
            print("Exiting. Thank you!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()