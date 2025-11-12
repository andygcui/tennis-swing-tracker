import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def play_video(video_path):
    """Play a video with pose tracking visualization."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000/fps)  # Delay between frames in milliseconds
    
    print(f"\nPlaying: {os.path.basename(video_path)}")
    print("Controls:")
    print("Space: Pause/Resume")
    print("Right Arrow: Next frame")
    print("Left Arrow: Previous frame")
    print("ESC: Exit")

    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                continue
            
            frame_count += 1
            
            # Convert BGR to RGB for pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Draw pose landmarks if detected
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
            
            # Draw frame number
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Tennis Pose Tracking', frame)
            
            # Wait for key press
            key = cv2.waitKey(frame_delay)
            
            if key == 27:  # ESC key
                break
            elif key == 32:  # Space key
                paused = True
            elif key == 83:  # Right arrow
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
            elif key == 81:  # Left arrow
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
        else:
            key = cv2.waitKey(0)
            if key == 32:  # Space key
                paused = False
            elif key == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Create necessary directories
    os.makedirs('database', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # List available videos in output directory
    output_dir = 'output'
    videos = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    
    if not videos:
        print("No processed videos found in the output directory.")
        return

    print("\nAvailable videos:")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video}")

    while True:
        try:
            choice = input("\nEnter the number of the video to play (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            choice = int(choice)
            if 1 <= choice <= len(videos):
                video_path = os.path.join(output_dir, videos[choice-1])
                play_video(video_path)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main() 