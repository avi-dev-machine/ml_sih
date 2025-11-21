
# ======================== IMPORTS ========================
import cv2                 
import os                   
import random               
import numpy as np
from ultralytics import YOLO 
from utils import (          
    align_points_to_fixed_reference_line,
    getAngle,
    draw_pts,
    draw_connection, 
    draw_pushup_bar,
    draw_counter
)

# Import your custom analyzer
from advanced_analyzer import (
    PushUpAnalyzer,
    draw_advanced_metrics,
    draw_real_time_feedback,
    print_advanced_metrics,
    print_athlete_performance,
)

# ======================== LOAD BACKGROUND IMAGE ========================
img_bg_path = 'bg.png'
img_bg = cv2.imread(img_bg_path)
if img_bg is None:
    img_bg = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img_bg, 'Menu: Press L/R/F for Video, C for Camera, S to Exit', 
                (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# ======================== LOAD YOLO MODEL ========================
model = YOLO('yolov8n-pose.pt')
print("Model device:", model.device)

# ======================== GLOBAL VARIABLES ========================
analyzer = PushUpAnalyzer() # Initialize the analyzer globally or per session
counter = 0
stage = 'up'

# ======================== HELPER FUNCTIONS ========================

def calculate_angle_coords(a, b, c):
    """
    Calculate angle between three points (a, b, c) where b is the vertex.
    Returns angle in degrees (float).
    """
    a = np.array(a) # First
    b = np.array(b) # Mid (Vertex)
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def process_side(frame, p, side_indices, relative_indices, connection_body, connection_relative,
                 anchor_idx, counter, stage, fixed_start, fixed_length, mode, pt1, pt2):
    """
    Process and draw alignment visualization (reference lines).
    Note: We rely on 'analyzer' for the actual counting logic now, 
    but this keeps your visual alignment features.
    """
    side_points = [p[i] for i in side_indices]
    relative_points = [p[i] for i in relative_indices]

    frame = draw_pts(frame, side_points)
    frame = draw_connection(frame, side_points, connection_body)

    # This getAngle call is for the visualization of the specific joint
    frame, angle_deg = getAngle(frame, p[anchor_idx], p[anchor_idx + 2], p[anchor_idx + 4])

    aligned_pts = align_points_to_fixed_reference_line(
        relative_points, pt1, pt2, frame, mode,
        fixed_start, fixed_length=fixed_length
    )

    frame = draw_pts(frame, aligned_pts[2:])
    frame = draw_connection(frame, aligned_pts, connection_relative)

    frame = draw_pushup_bar(frame, angle_deg)
    
    return frame, counter, stage


def get_random_video(folder_name):
    """Pick random video from folder"""
    candidates = [os.path.join('Videos', folder_name), os.path.join('example', folder_name)]
    folder_path = None
    for candidate in candidates:
        if os.path.isdir(candidate):
            folder_path = candidate
            break

    if folder_path is None:
        raise FileNotFoundError(f"No folder found for '{folder_name}'. Checked: {', '.join(candidates)}")

    video_files = [f for f in os.listdir(folder_path) if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))]
    if not video_files:
        raise FileNotFoundError(f"No video files found in '{folder_path}'")

    selected_video = random.choice(video_files)
    video_path = os.path.join(folder_path, selected_video)
    return video_path


def run_pose_detection(video_source, side, is_live=False):
    """Main pose detection loop"""
    global counter, stage, analyzer
    
    # Reset analyzer stats at the start of a new session
    analyzer = PushUpAnalyzer()
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_source}")
        return

    connection_body = [(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 6)]
    connection_relative = [(2, 3), (3, 4), (4, 5), (2, 5)]

    source_type = "LIVE CAMERA" if is_live else "VIDEO FILE"
    print(f"Starting {source_type} | Side: {side}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        
        # Add mode indicator
        mode_text = f"Mode: {source_type} | Side: {side}"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'S' to stop", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        results = model.predict(frame, conf=0.5, verbose=False)
        
        try:
            keypoints = results[0].keypoints.xy[0]
            p = keypoints.cpu().numpy()
        except Exception:
            cv2.imshow("PushUp Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('s') or key == ord('S'):
                break
            continue

        # ---------------------------------------------------------
        # 1. CALCULATE ANGLES FOR ANALYZER
        # ---------------------------------------------------------
        elbow_angle = 0
        shoulder_angle = 0
        hip_angle = 0
        
        # Determine which side to track for logic (Prefer Right if side is R, else Left)
        # Keypoint Map: 
        # Left: 5(Shoulder), 7(Elbow), 9(Wrist), 11(Hip), 13(Knee)
        # Right: 6(Shoulder), 8(Elbow), 10(Wrist), 12(Hip), 14(Knee)
        
        track_right = (side == 'R')
        
        if track_right:
            elbow_angle = calculate_angle_coords(p[6], p[8], p[10])     # R Shoulder-Elbow-Wrist
            shoulder_angle = calculate_angle_coords(p[8], p[6], p[12])  # R Elbow-Shoulder-Hip
            hip_angle = calculate_angle_coords(p[6], p[12], p[14])      # R Shoulder-Hip-Knee
        else:
            # Default to Left for 'L' and 'F' modes
            elbow_angle = calculate_angle_coords(p[5], p[7], p[9])      # L Shoulder-Elbow-Wrist
            shoulder_angle = calculate_angle_coords(p[7], p[5], p[11])  # L Elbow-Shoulder-Hip
            hip_angle = calculate_angle_coords(p[5], p[11], p[13])      # L Shoulder-Hip-Knee

        # ---------------------------------------------------------
        # 2. UPDATE ANALYZER & DRAW METRICS (Your Requested Code)
        # ---------------------------------------------------------
        data = analyzer.update(
            elbow_angle, 
            shoulder_angle, 
            hip_angle,
            left_elbow=p[7],      # Passed actual point
            right_elbow=p[8],     # Passed actual point
            pts_body=p            # Pass all keypoints
        )

        frame = draw_advanced_metrics(frame, analyzer, terminal=True)
        frame = draw_real_time_feedback(frame, data['feedback'])
        
        # Sync local variables with analyzer state
        counter = analyzer.counter
        stage = analyzer.stage

        # ---------------------------------------------------------
        # 3. DRAW ALIGNMENT VISUALS (Existing Logic)
        # ---------------------------------------------------------
        
        # Process LEFT side visuals
        if side in ['L', 'F']:
            pt1 = p[5]
            pt2 = p[15]
            frame, _, _ = process_side(
                frame, p,
                side_indices=[0, 5, 7, 9, 11, 13, 15],
                relative_indices=[0, 7, 5, 11, 13, 15],
                connection_body=connection_body,
                connection_relative=connection_relative,
                anchor_idx=5,
                counter=counter, 
                stage=stage,
                fixed_start=(690, 50),
                fixed_length=400,
                mode=side,
                pt1=pt1, 
                pt2=pt2,
            )

        # Process RIGHT side visuals
        if side in ['R', 'F']:
            pt1 = p[16]
            pt2 = p[6]
            frame, _, _ = process_side(
                frame, p, 
                side_indices=[0, 6, 8, 10, 12, 14, 16],
                relative_indices=[0, 8, 6, 12, 14, 16],
                connection_body=connection_body,
                connection_relative=connection_relative,
                anchor_idx=6,
                counter=counter, 
                stage=stage,
                fixed_start=(190, 50),
                fixed_length=400,
                mode=side,
                pt1=pt1, 
                pt2=pt2,
            )

        cv2.imshow("PushUp Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('s') or key == ord('S'):
            break
            
    # Session ended: print final session metrics and athlete performance
    print("\nSession ended. Final metrics:")
    print_advanced_metrics(analyzer)
    print_athlete_performance(analyzer)

    # Reset counter on exit
    counter = 0


# ======================== MAIN MENU ========================
print("\n" + "="*50)
print("      PUSH-UP COUNTER - LIVE & VIDEO MODE")
print("="*50)
print("\nControls:")
print("  L - Left Side (Video)")
print("  R - Right Side (Video)")
print("  F - Front View (Video)")
print("  C - Live Camera (Front View)")
print("  S - Exit")
print("\n" + "="*50 + "\n")

cv2.imshow("PushUp Counter", img_bg)

while True:
    key = cv2.waitKey(0) & 0xFF

    # VIDEO MODE - LEFT
    if key == ord('L') or key == ord('l'):
        try:
            video_path = get_random_video('L')
            print(f"\n[VIDEO LEFT] Playing: {video_path}")
            run_pose_detection(video_path, side='L', is_live=False)
            print("[VIDEO LEFT] Session ended\n")
        except Exception as e:
            print(f"Error: {e}\n")
        cv2.imshow("PushUp Counter", img_bg)

    # VIDEO MODE - RIGHT
    elif key == ord('R') or key == ord('r'):
        try:
            video_path = get_random_video('R')
            print(f"\n[VIDEO RIGHT] Playing: {video_path}")
            run_pose_detection(video_path, side='R', is_live=False)
            print("[VIDEO RIGHT] Session ended\n")
        except Exception as e:
            print(f"Error: {e}\n")
        cv2.imshow("PushUp Counter", img_bg)

    # VIDEO MODE - FRONT
    elif key == ord('F') or key == ord('f'):
        try:
            video_path = get_random_video('F')
            print(f"\n[VIDEO FRONT] Playing: {video_path}")
            run_pose_detection(video_path, side='F', is_live=False)
            print("[VIDEO FRONT] Session ended\n")
        except Exception as e:
            print(f"Error: {e}\n")
        cv2.imshow("PushUp Counter", img_bg)

    # LIVE CAMERA MODE
    elif key == ord('C') or key == ord('c'):
        print("\n[LIVE CAMERA] Starting webcam (0)...")
        print("Make sure your camera is connected and not in use by another app")
        try:
            run_pose_detection(0, side='F', is_live=True)  # 0 = default webcam
            print("[LIVE CAMERA] Session ended\n")
        except Exception as e:
            print(f"Error: {e}\n")
        cv2.imshow("PushUp Counter", img_bg)

    # EXIT
    elif key == 27 or key == ord('s') or key == ord('S'):
        print("\nExiting program...")
        break
    
    else:
        print("\nInvalid key! Press:")
        print("  L - Left Side (Video)")
        print("  R - Right Side (Video)")
        print("  F - Front View (Video)")
        print("  C - Live Camera (Front View)")
        print("  S - Exit\n")
        cv2.imshow("PushUp Counter", img_bg)

cv2.destroyAllWindows()
print("Program closed successfully!")
