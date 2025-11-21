import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

class MedicineBallThrowTest:
    def __init__(self):
        self.pose_model = YOLO("yolov8n-pose.pt")
        self.object_model = YOLO("yolov8n.pt")  # For ball detection
        self.reset_stats()
        # GUI / validation config (normalized thresholds)
        self.cfg = {
            'torso_offset_frac': 0.14,    # fraction of frame width (more forgiving)
            'shoulder_hip_ratio': 0.85,   # shoulder width < ratio * hip width => side-facing (more forgiving)
            'leg_x_offset_frac': 0.05,    # knee x must be ahead of hip x by this fraction of frame width
            'hold_distance_px': 120,      # px threshold for ball-to-wrist hold detection
            'ball_smooth_len': 5,
        }
        # UI state
        self.show_debug = False

    def reset_stats(self):
        self.throws = []  # list of throw attempts
        self.current_throw = None
        self.ball_trajectory = []  # track ball path
        self.max_distance = 0.0
        self.best_throw = None
        self.session_time = 0.0
        self.frame_count = 0
        self.throw_count = 0
        
        # Test validation tracking
        self.is_sitting = False
        self.back_against_wall = False
        self.chest_pass_form = False
        
        # Throw state machine
        self.is_holding = False
        self.is_releasing = False
        self.is_in_flight = False
        self.throw_start_frame = None
        self.release_point = None
        # Smoothing buffer for detected ball position to reduce jitter
        ball_smooth_len = getattr(self, 'cfg', {}).get('ball_smooth_len', 5)
        self.ball_smooth = deque(maxlen=ball_smooth_len)
        self.show_debug = False

    def kp_point(self, kpts, idx, conf_thresh=0.5):
        """Return (x,y) for keypoint idx if present and above confidence threshold.
        Handles kpts with shape (N,2) or (N,3).
        """
        if kpts is None:
            return None
        if idx < 0 or idx >= kpts.shape[0]:
            return None
        pt = kpts[idx]
        # Some builds return nested arrays or lists; guard against that
        if pt is None:
            return None
        # If pt has confidence channel
        try:
            if len(pt) >= 3:
                if float(pt[2]) > conf_thresh:
                    return (float(pt[0]), float(pt[1]))
                else:
                    return None
            else:
                # Only x,y available — accept the point if coordinates are positive
                if float(pt[0]) >= 0 or float(pt[1]) >= 0:
                    return (float(pt[0]), float(pt[1]))
                return None
        except Exception:
            return None

    def get_midpoint(self, kpts, idx1, idx2):
        """Calculate midpoint between two keypoints"""
        if kpts is None:
            return None

        p1 = self.kp_point(kpts, idx1)
        p2 = self.kp_point(kpts, idx2)

        if p1 and p2:
            return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        elif p1:
            return p1
        elif p2:
            return p2
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

    def detect_ball(self, frame, results):
        """Detect sports ball in frame (COCO class 32)"""
        ball_pos = None
        ball_conf = 0.0
        ball_box = None
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                # Class 32 = sports ball
                if cls == 32 and conf > 0.35:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    ball_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
                    ball_conf = conf
                    ball_box = (int(x1), int(y1), int(x2), int(y2))
                    break
        
        return ball_pos, ball_conf, ball_box

    def validate_seated_position(self, kpts, frame_height, frame_width):
        """
        Validate proper seated position against wall:
        - Hips should be low (seated)
        - Back relatively vertical (against wall)
        - Legs extended or slightly bent forward
        """
        # New, defensive seated-position check that enforces:
        #  - hips low (seated)
        #  - back roughly vertical / against wall (shoulder midpoint aligned with hip midpoint)
        #  - user is side-facing (foreshortening of shoulders vs hips)

        # Get visible keypoints (using safe kp_point accessor)
        left_sho = self.kp_point(kpts, 5)
        right_sho = self.kp_point(kpts, 6)
        left_hip = self.kp_point(kpts, 11)
        right_hip = self.kp_point(kpts, 12)
        left_knee = self.kp_point(kpts, 13)
        right_knee = self.kp_point(kpts, 14)
        left_ankle = self.kp_point(kpts, 15)
        right_ankle = self.kp_point(kpts, 16)

        # Require core keypoints for reliable check
        if not all([left_sho, right_sho, left_hip, right_hip]):
            debug = {
                'hip_ratio': None,
                'torso_offset': None,
                'shoulder_width': None,
                'hip_width': None,
                'side_facing': False,
                'legs_extended': False
            }
            return False, "Position unclear", debug

        # Midpoints (guarded)
        sho_mid = ((left_sho[0] + right_sho[0]) / 2.0, (left_sho[1] + right_sho[1]) / 2.0) if left_sho and right_sho else None
        hip_mid = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0) if left_hip and right_hip else None

        # 1) Seated check: hips in lower half of frame
        hip_height_ratio = (hip_mid[1] / frame_height) if hip_mid else 0.0
        is_seated = hip_height_ratio > 0.50

        # 2) Back against wall: shoulder-hip horizontal alignment (small x offset)
        torso_horiz_offset = abs(sho_mid[0] - hip_mid[0]) if sho_mid and hip_mid else float('inf')
        # threshold based on frame width; frame_w provided by caller via validate_seated_position earlier
        # fallback to constant if not provided
        fw = frame_width if frame_width is not None else 640
        back_against_wall = torso_horiz_offset < (self.cfg['torso_offset_frac'] * fw)

        # 3) Side-facing heuristic (foreshortening): shoulders appear narrower than hips when side-on
        shoulder_width = abs(left_sho[0] - right_sho[0]) if left_sho and right_sho else 0.0
        hip_width = abs(left_hip[0] - right_hip[0]) if left_hip and right_hip else 0.0
        side_facing = False
        if hip_width > 1:
            # If shoulder width is noticeably smaller than hip width -> side-on
            side_facing = shoulder_width < (self.cfg.get('shoulder_hip_ratio', 0.75) * hip_width)

        # 4) Legs roughly extended: check at least one knee/ankle is present and roughly in front
        legs_extended = False
        leg_offset_px = int(self.cfg.get('leg_x_offset_frac', 0.06) * (frame_width if frame_width else 640))
        if left_knee and left_hip:
            if abs(left_knee[0] - left_hip[0]) > leg_offset_px:
                legs_extended = True
        if right_knee and right_hip:
            if abs(right_knee[0] - right_hip[0]) > leg_offset_px:
                legs_extended = True

        debug = {
            'hip_ratio': hip_height_ratio,
            'torso_offset': torso_horiz_offset,
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'side_facing': side_facing,
            'legs_extended': legs_extended
        }

        # Compose validation messages
        if not is_seated:
            return False, "STAND UP - Must be seated", debug
        if not back_against_wall:
            return False, "LEAN BACK - Back must be against wall", debug
        if not side_facing:
            return False, "TURN SIDEWAYS - Face side to camera with back to wall", debug
        if not legs_extended:
            return False, "EXTEND LEGS - Legs forward", debug

        return True, "POSITION VALID", debug

    def validate_chest_pass_form(self, sho, elb_left, elb_right, wrist):
        """
        Validate proper chest pass technique:
        - Both elbows should be at similar height (symmetrical)
        - Arms should extend forward from chest
        - No overhead or underhand motion
        """
        if not all([sho, elb_left, elb_right, wrist]):
            return False, 0.0
        
        left_elb = (elb_left[0], elb_left[1])
        right_elb = (elb_right[0], elb_right[1])
        
        # Check elbow symmetry (both at similar height)
        elbow_height_diff = abs(left_elb[1] - right_elb[1])
        is_symmetric = elbow_height_diff < 50
        
        # Check elbows are at chest level (not too high or low)
        avg_elbow_y = (left_elb[1] + right_elb[1]) / 2
        chest_level = abs(avg_elbow_y - sho[1]) < 100
        
        # Check wrist extension (hands forward)
        wrist_forward = wrist[0] > sho[0]
        
        form_score = 100.0
        if not is_symmetric:
            form_score -= 30
        if not chest_level:
            form_score -= 30
        if not wrist_forward:
            form_score -= 20
        
        is_valid = form_score >= 60
        return is_valid, form_score

    def calculate_throw_distance(self, trajectory, release_point):
        """
        Calculate horizontal throw distance in pixels
        Distance = horizontal displacement from release point to landing
        """
        if not trajectory or len(trajectory) < 3 or not release_point:
            return 0.0
        
        # Release point X coordinate
        release_x = release_point[0]
        
        # Find landing point (furthest horizontal point)
        max_x = max(pt[0] for pt in trajectory)
        
        # Calculate horizontal distance
        distance = abs(max_x - release_x)
        
        return float(distance)

    def calculate_release_velocity(self, trajectory):
        """Calculate ball velocity at release (first few frames)"""
        if len(trajectory) < 3:
            return 0.0
        
        # Use first 3-5 frames after release
        early_trajectory = trajectory[:min(5, len(trajectory))]
        velocities = []
        
        for i in range(1, len(early_trajectory)):
            dx = early_trajectory[i][0] - early_trajectory[i-1][0]
            dy = early_trajectory[i][1] - early_trajectory[i-1][1]
            vel = np.sqrt(dx**2 + dy**2)
            velocities.append(vel)
        
        return float(np.mean(velocities)) if velocities else 0.0

    def process_source(self, source_path=None, is_webcam=True):
        cap_arg = 0 if source_path is None else source_path
        cap = cv2.VideoCapture(cap_arg)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        frame_duration = 1.0 / fps
        
        window_name = "Medicine Ball Throw Test (Seated Chest Pass)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n--- MEDICINE BALL THROW TEST ---")
        print("INSTRUCTIONS:")
        print("  1. Sit on floor with back against wall")
        print("  2. Hold medicine ball at chest level")
        print("  3. Throw forward using chest pass motion")
        print("  4. Keep back against wall during throw")
        print("\nPress 'R' to reset attempt | 'Q' to quit\n")
        
        frames_without_ball = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if is_webcam: frame = cv2.flip(frame, 1)

            # Run both detection models
            pose_results = self.pose_model(frame, verbose=False)
            object_results = self.object_model(frame, verbose=False)
            
            overlay = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            
            status = "Position Yourself"
            color = (200, 200, 200)
            position_valid = False
            form_valid = False
            
            # Detect ball
            ball_pos, ball_conf, ball_box = self.detect_ball(frame, object_results)

            # Smooth ball position to reduce detection jitter
            smooth_ball = None
            if ball_pos is not None:
                try:
                    self.ball_smooth.append((float(ball_pos[0]), float(ball_pos[1])))
                    arr = np.array(self.ball_smooth)
                    smooth_ball = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
                except Exception:
                    smooth_ball = (float(ball_pos[0]), float(ball_pos[1]))
            else:
                # do not clear smooth buffer immediately; allow short occlusions
                if len(self.ball_smooth) == 0:
                    smooth_ball = None
                else:
                    arr = np.array(self.ball_smooth)
                    smooth_ball = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
            
            # Process athlete pose
            if pose_results[0].keypoints is not None and pose_results[0].keypoints.xy.numel() > 0:
                kpts = pose_results[0].keypoints.xy[0].cpu().numpy()
                
                # Key body points (use safe accessor for individual points)
                sho = self.get_midpoint(kpts, 5, 6)
                elb_left = self.kp_point(kpts, 7)
                elb_right = self.kp_point(kpts, 8)
                elb = self.get_midpoint(kpts, 7, 8)
                wrist = self.get_midpoint(kpts, 9, 10)
                hip = self.get_midpoint(kpts, 11, 12)
                knee = self.get_midpoint(kpts, 13, 14)
                ankle = self.get_midpoint(kpts, 15, 16)
                
                # Draw skeleton
                if sho and hip and knee and ankle:
                    pt_sho = (int(sho[0]), int(sho[1]))
                    pt_hip = (int(hip[0]), int(hip[1]))
                    pt_knee = (int(knee[0]), int(knee[1]))
                    pt_ankle = (int(ankle[0]), int(ankle[1]))
                    
                    # Torso (back against wall emphasis)
                    cv2.line(overlay, pt_sho, pt_hip, (255, 255, 255), 3)
                    # Legs (extended)
                    cv2.line(overlay, pt_hip, pt_knee, (255, 255, 255), 2)
                    cv2.line(overlay, pt_knee, pt_ankle, (255, 255, 255), 2)
                    
                    # Arms (chest pass emphasis)
                    if elb and wrist:
                        pt_elb = (int(elb[0]), int(elb[1]))
                        pt_wrist = (int(wrist[0]), int(wrist[1]))
                        cv2.line(overlay, pt_sho, pt_elb, (0, 255, 0), 3)
                        cv2.line(overlay, pt_elb, pt_wrist, (0, 255, 0), 3)
                        cv2.circle(overlay, pt_wrist, 10, (255, 0, 255), -1)
                    
                    # Keypoint markers
                    cv2.circle(overlay, pt_sho, 6, (0, 255, 255), -1)
                    cv2.circle(overlay, pt_hip, 10, (0, 0, 255), -1)
                    cv2.circle(overlay, pt_knee, 6, (0, 255, 255), -1)
                    cv2.circle(overlay, pt_ankle, 6, (0, 255, 255), -1)

                # Validate seated position
                position_valid, position_msg, position_debug = self.validate_seated_position(kpts, frame_h, frame_w)
                
                if not position_valid:
                    status = position_msg
                    color = (0, 0, 255)
                # Draw debug info (small values to help tune)
                if self.show_debug:
                    try:
                        dbg = position_debug
                        dbg_y = 80
                        dbg_x = frame_w - 260
                        cv2.rectangle(overlay, (dbg_x - 10, dbg_y - 20), (frame_w - 10, dbg_y + 140), (0,0,0), -1)
                        cv2.putText(overlay, f"HipRatio:{dbg.get('hip_ratio'):.2f}" if dbg.get('hip_ratio') is not None else "HipRatio: N/A", (dbg_x, dbg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        cv2.putText(overlay, f"TorsoOff:{dbg.get('torso_offset'):.1f}", (dbg_x, dbg_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        cv2.putText(overlay, f"ShoWid:{dbg.get('shoulder_width'):.1f}", (dbg_x, dbg_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        cv2.putText(overlay, f"HipWid:{dbg.get('hip_width'):.1f}", (dbg_x, dbg_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        cv2.putText(overlay, f"SideFacing:{dbg.get('side_facing')}", (dbg_x, dbg_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        cv2.putText(overlay, f"LegsExt:{dbg.get('legs_extended')}", (dbg_x, dbg_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    except Exception:
                        pass
                else:
                    # Validate chest pass form
                    if elb_left and elb_right:
                        left_elb_pt = (elb_left[0], elb_left[1])
                        right_elb_pt = (elb_right[0], elb_right[1])
                        form_valid, form_score = self.validate_chest_pass_form(
                            sho, left_elb_pt, right_elb_pt, wrist
                        )
                    else:
                        form_valid = False
                        form_score = 0.0
                    
                    # Calculate arm extension (for throw detection)
                    elbow_angle = 0
                    if sho and elb and wrist:
                        elbow_angle = self.calculate_angle(sho, elb, wrist)
                    
                    # Throw state machine
                    if smooth_ball and wrist:
                        ball_to_wrist = self.calculate_distance(smooth_ball, wrist)
                        
                        # State 1: Holding ball at chest
                        hold_th = self.cfg.get('hold_distance_px', 120)
                        if ball_to_wrist < hold_th and not self.is_in_flight:
                            self.is_holding = True
                            status = "READY - Hold at chest"
                            color = (0, 255, 255)
                            
                            if self.current_throw is None:
                                self.current_throw = {
                                    'start_time': self.session_time,
                                    'start_frame': self.frame_count,
                                    'position_valid': position_valid,
                                    'form_score': form_score,
                                }
                        
                        # State 2: Release detected (ball leaving hands)
                        elif ball_to_wrist is not None and ball_to_wrist > hold_th and self.is_holding and not self.is_in_flight:
                            self.is_releasing = True
                            self.is_holding = False
                            self.is_in_flight = True
                            self.release_point = smooth_ball
                            self.ball_trajectory = [smooth_ball]
                            
                            status = "RELEASING"
                            color = (0, 255, 0)
                            
                            if self.current_throw:
                                self.current_throw['release_point'] = self.release_point
                                self.current_throw['release_elbow_angle'] = float(elbow_angle)
                                self.current_throw['release_frame'] = self.frame_count
                    
                    # State 3: Ball in flight
                    if self.is_in_flight and smooth_ball:
                        self.ball_trajectory.append(smooth_ball)
                        status = "IN FLIGHT - Tracking"
                        color = (255, 165, 0)
                        frames_without_ball = 0
                    elif self.is_in_flight:
                        frames_without_ball += 1
                    
                    # State 4: Landing detected (ball disappeared)
                    if self.is_in_flight and frames_without_ball > 12:
                        self.is_in_flight = False
                        
                        if self.current_throw and len(self.ball_trajectory) >= 5:
                            # Calculate throw distance
                            distance = self.calculate_throw_distance(
                                self.ball_trajectory, self.release_point
                            )
                            
                            # Calculate release velocity
                            release_vel = self.calculate_release_velocity(self.ball_trajectory)
                            
                            # Calculate trajectory quality
                            arc_quality = 100.0
                            if len(self.ball_trajectory) > 3:
                                y_coords = [pt[1] for pt in self.ball_trajectory]
                                arc_quality = max(0.0, 100.0 - np.std(np.diff(y_coords)) * 2)
                            
                            # Complete throw record
                            self.current_throw['end_time'] = self.session_time
                            self.current_throw['distance_px'] = distance
                            self.current_throw['release_velocity'] = release_vel
                            self.current_throw['arc_quality'] = float(arc_quality)
                            self.current_throw['trajectory_length'] = len(self.ball_trajectory)
                            self.current_throw['valid_attempt'] = (
                                position_valid and form_valid and distance > 50
                            )
                            
                            # Update best throw
                            if distance > self.max_distance:
                                self.max_distance = distance
                                self.best_throw = self.current_throw.copy()
                            
                            # Save throw
                            self.throws.append(self.current_throw)
                            self.throw_count += 1
                            
                            # Print immediate feedback
                            print(f"\n{'='*50}")
                            print(f"THROW #{self.throw_count} RECORDED")
                            print(f"{'='*50}")
                            print(f"  Distance: {distance:.1f} pixels")
                            print(f"  Release Velocity: {release_vel:.1f} px/frame")
                            print(f"  Form Score: {self.current_throw['form_score']:.1f}%")
                            print(f"  Position Valid: {'✓' if position_valid else '✗'}")
                            print(f"  Valid Attempt: {'✓' if self.current_throw['valid_attempt'] else '✗'}")
                            print(f"{'='*50}\n")
                        
                        # Reset for next throw
                        self.current_throw = None
                        self.ball_trajectory = []
                        self.is_holding = False
                        self.is_releasing = False
                        self.release_point = None
                        frames_without_ball = 0

            # Draw ball detection
            # Draw smoothed ball (if any) and label clearly
            if smooth_ball is not None:
                bx, by = int(smooth_ball[0]), int(smooth_ball[1])
                cv2.circle(overlay, (bx, by), 18, (0, 150, 200), -1)
                label = f"Medicine Ball {ball_conf:.2f}"
                text_pos = (min(frame_w - 10, bx + 25), max(10, by - 10))
                # draw text outline for readability
                cv2.putText(overlay, label, (text_pos[0], text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(overlay, label, (text_pos[0], text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 70), 1)

                if ball_box:
                    # Draw bounding box and label above it
                    x1, y1, x2, y2 = ball_box
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w - 1, x2), min(frame_h - 1, y2)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 150, 200), 2)
                    lbl_pos = (x1, max(10, y1 - 10))
                    cv2.putText(overlay, label, (lbl_pos[0], lbl_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                    cv2.putText(overlay, label, (lbl_pos[0], lbl_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,70), 1)
            
                # Draw ball trajectory
            if len(self.ball_trajectory) > 1:
                for i in range(1, min(len(self.ball_trajectory), 40)):
                    pt1 = (int(self.ball_trajectory[-i-1][0]), int(self.ball_trajectory[-i-1][1]))
                    pt2 = (int(self.ball_trajectory[-i][0]), int(self.ball_trajectory[-i][1]))
                    thickness = max(1, 4 - i // 10)
                    cv2.line(overlay, pt1, pt2, (255, 0, 255), thickness)
            
            # Update session time
            self.session_time += frame_duration
            self.frame_count += 1
            
            # Clean header + metrics (no large black box)
            header_font = cv2.FONT_HERSHEY_DUPLEX
            small_font = cv2.FONT_HERSHEY_SIMPLEX
            accent_green = (34, 177, 76)
            accent_orange = (0, 165, 255)
            accent_blue = (50, 150, 255)

            # Header (top-center)
            header_text = "Medicine Ball Throw - Seated Chest Pass"
            (w_text, h_text), _ = cv2.getTextSize(header_text, header_font, 0.8, 1)
            cv2.putText(overlay, header_text, ((frame_w - w_text)//2, 28), header_font, 0.8, (230,230,230), 1)

            # Metrics (left side, compact)
            mx = 12
            my = 48
            line_h = 26
            valid_throws = sum(1 for t in self.throws if t.get('valid_attempt', False))
            cv2.putText(overlay, f"Attempts: {self.throw_count}", (mx, my), small_font, 0.7, (240,240,240), 1)
            cv2.putText(overlay, f"Valid: {valid_throws}", (mx, my + line_h), small_font, 0.7, accent_green if valid_throws>0 else (180,180,180), 1)
            cv2.putText(overlay, f"Best: {self.max_distance:.0f}px", (mx, my + 2*line_h), small_font, 0.7, accent_blue, 1)

            # Status (right side)
            status_x = frame_w - 260
            cv2.putText(overlay, status, (status_x, 48), small_font, 0.8, color, 2)
            pos_text = "Position: VALID" if position_valid else "Position: INVALID"
            pos_color = accent_green if position_valid else (0,0,255)
            cv2.putText(overlay, pos_text, (status_x, 48 + line_h), small_font, 0.7, pos_color, 2)

            # Instructions (bottom-left)
            instr_y = frame_h - 16
            cv2.putText(overlay, "R: Reset | D: Toggle Debug | Q: Quit", (12, instr_y), small_font, 0.6, (220,220,220), 1)

            cv2.imshow(window_name, overlay)
            
            delay = 1 if is_webcam else int(1000/fps)
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset current attempt
                self.current_throw = None
                self.ball_trajectory = []
                self.is_holding = False
                self.is_releasing = False
                self.is_in_flight = False
                self.release_point = None
                frames_without_ball = 0
                print("\n⟳ Attempt reset!\n")
            elif key == ord('d'):
                # Toggle debug panel
                self.show_debug = not self.show_debug
                print(f"Debug panel {'ON' if self.show_debug else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.print_statistics()

    def print_statistics(self):
        import statistics, json, os

        print("\n" + "="*70)
        print("MEDICINE BALL THROW TEST - PERFORMANCE REPORT")
        print("="*70)
        
        if not self.throws:
            print("\nNo throws recorded.")
            return
        
        # Separate valid and invalid attempts
        valid_throws = [t for t in self.throws if t.get('valid_attempt', False)]
        invalid_throws = [t for t in self.throws if not t.get('valid_attempt', False)]
        
        print(f"\nTotal Attempts: {len(self.throws)}")
        print(f"Valid Throws: {len(valid_throws)}")
        print(f"Invalid Throws: {len(invalid_throws)}")
        
        if not valid_throws:
            print("\n⚠ No valid throws recorded. Ensure proper seated position.")
            return
        
        # Extract metrics from valid throws only
        distances = [t['distance_px'] for t in valid_throws]
        velocities = [t['release_velocity'] for t in valid_throws]
        form_scores = [t['form_score'] for t in valid_throws]
        arc_qualities = [t['arc_quality'] for t in valid_throws]
        
        # Primary metric: Best throw distance
        best_distance = max(distances)
        avg_distance = float(np.mean(distances))
        
        print(f"\n{'='*70}")
        print("PRIMARY MEASUREMENT: THROW DISTANCE")
        print(f"{'='*70}")
        print(f"  Best Throw: {best_distance:.1f} pixels")
        print(f"  Average: {avg_distance:.1f} pixels")
        print(f"  Range: {min(distances):.1f} - {max(distances):.1f} pixels")
        
        if len(distances) > 1:
            consistency = float(np.std(distances))
            cv = (consistency / avg_distance) * 100 if avg_distance > 0 else 0
            print(f"  Consistency: ±{consistency:.1f}px (CV: {cv:.1f}%)")
        
        # Secondary metrics
        print(f"\n{'='*70}")
        print("EXPLOSIVE POWER INDICATORS")
        print(f"{'='*70}")
        print(f"  Max Release Velocity: {max(velocities):.1f} px/frame")
        print(f"  Avg Release Velocity: {np.mean(velocities):.1f} px/frame")
        
        # Form and technique quality
        print(f"\n{'='*70}")
        print("TECHNIQUE & FORM QUALITY")
        print(f"{'='*70}")
        print(f"  Avg Form Score: {np.mean(form_scores):.1f}%")
        print(f"  Avg Arc Quality: {np.mean(arc_qualities):.1f}%")
        
        # Calculate overall performance score
        # Based on: distance (50%), velocity (25%), form (15%), consistency (10%)
        distance_score = min(100.0, (best_distance / 500.0) * 100)
        velocity_score = min(100.0, (max(velocities) / 60.0) * 100)
        form_score_avg = float(np.mean(form_scores))
        consistency_score = max(0.0, 100.0 - cv if len(distances) > 1 else 90.0)
        
        overall = (distance_score * 0.50 + velocity_score * 0.25 + 
                  form_score_avg * 0.15 + consistency_score * 0.10)
        overall = float(max(0.0, min(100.0, overall)))
        
        print(f"\n{'='*70}")
        print(f"OVERALL UPPER-BODY POWER SCORE: {overall:.1f}/100")
        print(f"{'='*70}")
        
        # Individual throw breakdown
        print(f"\n{'='*70}")
        print("VALID THROW DETAILS")
        print(f"{'='*70}")
        for i, throw in enumerate(valid_throws, 1):
            print(f"\nThrow {i}:")
            print(f"  Distance: {throw['distance_px']:.1f}px")
            print(f"  Release Velocity: {throw['release_velocity']:.1f}")
            print(f"  Form Score: {throw['form_score']:.1f}%")
            print(f"  Arc Quality: {throw['arc_quality']:.1f}%")
        
        if invalid_throws:
            print(f"\n{'='*70}")
            print("INVALID ATTEMPTS (Not counted in scoring)")
            print(f"{'='*70}")
            for i, throw in enumerate(invalid_throws, 1):
                print(f"  Attempt {i}: Position/form violation")
        
        # Export JSON report
        try:
            report = {
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'Medicine Ball Throw Test (Seated Chest Pass)',
                'session_duration_sec': float(self.session_time),
                'total_attempts': len(self.throws),
                'valid_throws': len(valid_throws),
                'invalid_throws': len(invalid_throws),
                'primary_measurement': {
                    'best_distance_px': float(best_distance),
                    'average_distance_px': avg_distance,
                    'consistency_std_dev': float(consistency) if len(distances) > 1 else None
                },
                'power_metrics': {
                    'max_release_velocity': float(max(velocities)),
                    'avg_release_velocity': float(np.mean(velocities))
                },
                'technique_metrics': {
                    'avg_form_score': float(np.mean(form_scores)),
                    'avg_arc_quality': float(np.mean(arc_qualities))
                },
                'performance_score': {
                    'distance_score': float(distance_score),
                    'velocity_score': float(velocity_score),
                    'form_score': form_score_avg,
                    'consistency_score': float(consistency_score),
                    'overall_score': overall
                },
                'valid_throws_data': valid_throws,
                'best_throw': self.best_throw
            }
            
            out_path = os.path.join(os.path.dirname(__file__) or '.', 
                                   f'medicine_ball_test_{int(time.time())}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f'\n✓ Detailed report exported to: {out_path}')
        except Exception as e:
            print(f'✗ Failed to export report: {e}')

def main():
    tracker = MedicineBallThrowTest()
    print("\n" + "="*70)
    print("MEDICINE BALL THROW TEST")
    print("Upper-Body Explosive Power Assessment")
    print("="*70)
    print("\nTEST PROTOCOL:")
    print("  • Sit on floor with back AGAINST WALL")
    print("  • Legs extended or slightly bent forward")
    print("  • Hold medicine ball at CHEST level with both hands")
    print("  • Use CHEST PASS motion to throw forward as far as possible")
    print("  • Keep back against wall during entire throw")
    print("  • No trunk lean or momentum from upper body")
    print("\nSCORING:")
    print("  • Distance is measured from release point to landing")
    print("  • Best of 2-3 valid attempts is recorded")
    print("  • Invalid throws (wall separation, poor form) are not counted")
    print("\nREQUIREMENTS:")
    print("  • Medicine ball (2-5kg recommended)")
    print("  • Bright colored ball for better detection")
    print("  • Clear background and good lighting")
    print("\nPress 'R' to reset attempt | 'Q' to finish test\n")
    # Interactive launcher: choose webcam or video file
    while True:
        choice = input("\nW: Webcam | V: Video File | Q: Quit > ").strip().lower()
        if choice == 'w':
            tracker.reset_stats()
            try:
                tracker.process_source(0, True)
            except Exception as e:
                print(f"Error running webcam: {e}")
        elif choice == 'v':
            video_path = input("Enter video filename: ").strip()
            if not video_path:
                print("No video path provided.")
                continue
            tracker.reset_stats()
            try:
                tracker.process_source(video_path, False)
            except Exception as e:
                print(f"Error processing video: {e}")
        elif choice == 'q':
            print("Exiting. Thank you!")
            break
        else:
            print("Invalid choice. Please enter W, V, or Q.")
    

if __name__ == "__main__":
    main()