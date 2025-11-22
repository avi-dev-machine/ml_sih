import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO


class PoseDetector:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.results = None
        self.img_shape = None

    def find_pose(self, img, draw=True):
        """Detect pose and optionally draw skeleton"""
        if self.img_shape is None:
            self.img_shape = img.shape

        self.results = self.model(img, verbose=False)
        
        if draw and self.results[0].keypoints is not None:
            img = self.draw_skeleton(img)
        
        return img

    def draw_skeleton(self, img):
        """Draw full skeleton with connections"""
        if self.results[0].keypoints.xy.numel() == 0:
            return img
            
        kpts = self.results[0].keypoints.xy[0].cpu().numpy()
        
        # Skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(kpts) and end_idx < len(kpts):
                start_pt = kpts[start_idx]
                end_pt = kpts[end_idx]
                
                if start_pt[0] > 0 and start_pt[1] > 0 and end_pt[0] > 0 and end_pt[1] > 0:
                    cv2.line(img, 
                            (int(start_pt[0]), int(start_pt[1])),
                            (int(end_pt[0]), int(end_pt[1])),
                            (255, 0, 0), 2)

        # Draw keypoints - highlight ankles
        for i, kpt in enumerate(kpts):
            if kpt[0] > 0 and kpt[1] > 0:
                color = (0, 255, 0) if i in [15, 16] else (255, 0, 0)
                radius = 6 if i in [15, 16] else 4
                cv2.circle(img, (int(kpt[0]), int(kpt[1])), radius, color, -1)

        return img

    def get_landmarks(self):
        """Extract landmark coordinates"""
        if self.results and self.results[0].keypoints is not None:
            if self.results[0].keypoints.xy.numel() > 0:
                h, w, c = self.img_shape
                kpts = self.results[0].keypoints.xy[0].cpu().numpy()
                return np.multiply(kpts, [1, 1]).astype(int)
        return []

    def get_avg_visibility(self):
        """Get average visibility of all keypoints"""
        if self.results and self.results[0].keypoints is not None:
            if self.results[0].keypoints.conf is not None:
                conf = self.results[0].keypoints.conf[0].cpu().numpy()
                return np.mean(conf)
        return 0.0

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return int(np.degrees(angle))

    def is_jumping(self, lm):
        """Detect if person is in jumping position - just check if person is visible"""
        # For jump rope, we don't need strict arm position checking
        # Just verify the person is detected and key points are visible
        if len(lm) < 17:
            return False
        
        # Check if ankles are detected (most important for jump detection)
        l_ankle = lm[15]
        r_ankle = lm[16]
        
        if l_ankle[0] > 0 and l_ankle[1] > 0 and r_ankle[0] > 0 and r_ankle[1] > 0:
            return True
        
        return False

    def count_jumping(self, img, count, is_jumping, base, position):
        """
        Count jumps by detecting when ankles lift off the ground.
        Simple and reliable detection method.
        """
        h, w, c = self.img_shape
        lm = self.get_landmarks()

        # Lift threshold - how much ankles need to lift to count as jump
        lift_threshold = int(h * 0.08)  # 8% of frame height

        if len(lm) != 0:
            # Get ankle positions (15: left ankle, 16: right ankle)
            l_ankle = lm[15]
            r_ankle = lm[16]
            
            # Get average ankle height
            avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2

            # Check if person is visible
            if self.get_avg_visibility() > 0.6 and self.is_jumping(lm):
                # Set baseline (ground position) when person first detected
                if not is_jumping:
                    base = avg_ankle_y
                    is_jumping = True
                    position = 0

                # Calculate detection line
                detection_line = base - lift_threshold

                # Simple state machine:
                # position 0 = on ground, waiting for jump
                # position 1 = in air, waiting to land
                
                # Ankles lifted above threshold - JUMP!
                if avg_ankle_y < detection_line and position == 0:
                    count += 1
                    position = 1
                    print(f"✓ Jump #{count} detected!")
                
                # Ankles returned to ground
                elif avg_ankle_y > detection_line and position == 1:
                    position = 0

                # Draw visualization
                img = self.show_jump_lines(img, lm, base, lift_threshold, 0, w, count)
            else:
                # Person not detected properly, reset
                if is_jumping:
                    is_jumping = False
                base = avg_ankle_y
                position = 0
        else:
            base = 0

        return img, count, is_jumping, base, position

    def show_jump_lines(self, img, lm, base, lift_threshold, unused, w, count):
        """Visualize jump detection line and ankle positions"""
        l_ankle = lm[15]
        r_ankle = lm[16]
        
        # Calculate average ankle position
        avg_ankle_y = int((l_ankle[1] + r_ankle[1]) / 2)

        # Draw ground baseline (white dashed line)
        for x in range(0, w, 20):
            cv2.line(img, (x, int(base)), (x + 10, int(base)), (255, 255, 255), 2)

        # Draw detection line (bright green)
        detection_line = int(base - lift_threshold)
        cv2.line(img, (0, detection_line), (w, detection_line), (0, 255, 0), 3)
        cv2.putText(img, "JUMP LINE", (10, detection_line - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw line from average ankle to show current position
        cv2.line(img, (w//2 - 50, avg_ankle_y), (w//2 + 50, avg_ankle_y), (255, 0, 255), 2)

        # Highlight ankles based on which is higher
        ankle_color_l = (0, 255, 0) if l_ankle[1] < r_ankle[1] else (255, 255, 255)
        ankle_color_r = (0, 255, 0) if r_ankle[1] < l_ankle[1] else (255, 255, 255)
        
        cv2.circle(img, (l_ankle[0], l_ankle[1]), 10, ankle_color_l, -1)
        cv2.circle(img, (r_ankle[0], r_ankle[1]), 10, ankle_color_r, -1)

        # Show if ankles are above detection line
        if avg_ankle_y < detection_line:
            cv2.putText(img, "LIFTED!", (w//2 - 60, detection_line - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        return img

    def jumping_speed(self, history, current_time, count):
        """Calculate jumps per minute over a rolling window"""
        buffer = 5  # seconds
        history = np.roll(history, -2)
        history[-1] = np.array([current_time, count])
        
        jumps_since = 0
        for i in history:
            if i[0] >= (current_time - buffer):
                jumps_since = count - i[1]
                break

        speed = jumps_since * (60 / buffer)  # per minute
        return history, speed


class JumpRopeCounter:
    def __init__(self):
        self.detector = PoseDetector()
        self.vidcap = None
        self.init_specs()

    def init_specs(self):
        """Initialize/reset all tracking variables"""
        self.start_time = time.time()
        self.previous_time = time.time()
        self.jumping_time = 0
        self.break_time = 0
        self.count = 0
        self.is_jumping = False
        self.position = 0
        self.baseline = 0.5
        self.history_list = np.zeros((50, 2))
        self.speed = 0
        self.speed_list = []
        self.max_speed = 0
        self.frame_count = 0
        self.fps = 0

    def format_time(self, seconds):
        """Format seconds as MM:SS"""
        return time.strftime("%M:%S", time.gmtime(seconds))

    def process_source(self, source_path=0, is_webcam=True):
        """Main processing loop"""
        self.vidcap = cv2.VideoCapture(source_path)
        
        if not self.vidcap.isOpened():
            print("Error: Could not open video source")
            return

        fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): 
            fps = 30

        window_name = "Jump Rope Counter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"\n{'='*60}")
        print(f"JUMP ROPE COUNTER")
        print(f"{'='*60}")
        print("Controls:")
        print("  Q - Quit and save session")
        print("  R - Reset counter")
        print("  P - Pause/Resume")
        print(f"{'='*60}\n")

        paused = False
        pause_start = None

        while self.vidcap.isOpened():
            if not paused:
                ret, frame = self.vidcap.read()
                if not ret:
                    if not is_webcam:
                        # Loop video
                        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                if is_webcam:
                    frame = cv2.flip(frame, 1)

                # Process frame
                frame = self.detector.find_pose(frame, draw=True)
                (
                    frame,
                    self.count,
                    self.is_jumping,
                    self.baseline,
                    self.position,
                ) = self.detector.count_jumping(
                    frame,
                    self.count,
                    self.is_jumping,
                    self.baseline,
                    self.position,
                )

                # Update times
                current_time = time.time()
                delta_time = current_time - self.previous_time
                
                if self.is_jumping:
                    self.jumping_time += delta_time
                
                self.previous_time = current_time
                total_time = current_time - self.start_time
                break_time = total_time - self.jumping_time

                # Calculate speed every few frames
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    self.history_list, self.speed = self.detector.jumping_speed(
                        self.history_list, total_time, self.count
                    )
                    self.speed_list.append([total_time, self.speed])
                    if self.speed > self.max_speed:
                        self.max_speed = self.speed

                # Calculate FPS
                self.fps = self.frame_count / total_time if total_time > 0 else 0

                # Draw dashboard
                self.draw_dashboard(frame, total_time, break_time)

            else:
                # Show paused message
                frame_copy = frame.copy()
                h, w = frame_copy.shape[:2]
                cv2.rectangle(frame_copy, (0, 0), (w, 100), (0, 0, 0), -1)
                cv2.putText(frame_copy, "PAUSED - Press 'P' to resume", 
                           (w//2 - 300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.imshow(window_name, frame_copy)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.save_session()
                break
            elif key == ord('r'):
                self.init_specs()
                print("\n--- Counter Reset ---\n")
            elif key == ord('p'):
                paused = not paused
                if paused:
                    pause_start = time.time()
                    print("Session paused")
                else:
                    if pause_start:
                        pause_duration = time.time() - pause_start
                        self.start_time += pause_duration
                        self.previous_time = time.time()
                    print("Session resumed")

            if not paused:
                cv2.imshow(window_name, frame)

        self.vidcap.release()
        cv2.destroyAllWindows()
        self.print_statistics()

    def draw_dashboard(self, img, total_time, break_time):
        """Draw statistics dashboard on frame"""
        h, w = img.shape[:2]
        
        # Main stats panel (top-left)
        panel_h = 200
        cv2.rectangle(img, (0, 0), (550, panel_h), (0, 0, 0), -1)
        
        # Jump count (large)
        cv2.putText(img, f"Jumps: {self.count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        
        # Current speed
        speed_color = (0, 255, 255) if self.speed > 0 else (100, 100, 100)
        cv2.putText(img, f"Speed: {int(self.speed)} JPM", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, speed_color, 2)
        
        # Max speed
        cv2.putText(img, f"Max: {int(self.max_speed)} JPM", (20, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status indicator
        status = "JUMPING" if self.is_jumping else "READY"
        status_color = (0, 255, 0) if self.is_jumping else (0, 0, 255)
        cv2.putText(img, status, (20, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Time panel (top-right)
        time_w = 300
        cv2.rectangle(img, (w - time_w, 0), (w, 150), (0, 0, 0), -1)
        
        cv2.putText(img, "Total Time:", (w - time_w + 20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(img, self.format_time(total_time), (w - time_w + 20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(img, f"Active: {self.format_time(self.jumping_time)}", 
                   (w - time_w + 20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 190, 0), 2)
        cv2.putText(img, f"Break: {self.format_time(break_time)}", 
                   (w - time_w + 20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 190, 0), 2)
        
        # FPS indicator (bottom-right)
        cv2.putText(img, f"FPS: {self.fps:.1f}", (w - 150, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def save_session(self):
        """Save session data to JSON file"""
        total_time = time.time() - self.start_time
        break_time = total_time - self.jumping_time
        
        session_data = {
            'total_jumps': self.count,
            'total_time': total_time,
            'jumping_time': self.jumping_time,
            'break_time': break_time,
            'max_speed': self.max_speed,
            'avg_speed': np.mean([s[1] for s in self.speed_list]) if self.speed_list else 0,
            'speed_history': self.speed_list,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            filename = f'jumprope_session_{int(time.time())}.json'
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            print(f'\n✓ Session saved to: {filepath}')
        except Exception as e:
            print(f'\n✗ Failed to save session: {e}')

    def print_statistics(self):
        """Print final session statistics"""
        total_time = time.time() - self.start_time
        break_time = total_time - self.jumping_time
        
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Jumps:     {self.count}")
        print(f"Total Time:      {self.format_time(total_time)}")
        print(f"Active Time:     {self.format_time(self.jumping_time)}")
        print(f"Break Time:      {self.format_time(break_time)}")
        print(f"Max Speed:       {int(self.max_speed)} jumps/min")
        
        if self.speed_list:
            avg_speed = np.mean([s[1] for s in self.speed_list])
            print(f"Average Speed:   {int(avg_speed)} jumps/min")
        
        if self.jumping_time > 0:
            efficiency = (self.jumping_time / total_time) * 100
            print(f"Efficiency:      {efficiency:.1f}%")
        
        print(f"{'='*60}\n")


def main():
    counter = JumpRopeCounter()
    
    while True:
        print("\n" + "="*60)
        print("JUMP ROPE COUNTER")
        print("="*60)
        choice = input("\nW: Webcam | V: Video | Q: Quit > ").lower()
        
        if choice == 'w':
            counter.init_specs()
            counter.process_source(0, True)
        elif choice == 'v':
            filename = input("Video filename: ")
            try:
                counter.init_specs()
                counter.process_source(filename, False)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()