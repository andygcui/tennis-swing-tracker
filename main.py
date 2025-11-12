import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class TennisTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Key points for tennis swing
        self.key_points = {
            'nose': [self.mp_pose.PoseLandmark.NOSE],
            'right_shoulder': [self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'left_shoulder': [self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            'right_elbow': [self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            'left_elbow': [self.mp_pose.PoseLandmark.LEFT_ELBOW],
            'right_wrist': [self.mp_pose.PoseLandmark.RIGHT_WRIST],
            'left_wrist': [self.mp_pose.PoseLandmark.LEFT_WRIST],
            'right_hip': [self.mp_pose.PoseLandmark.RIGHT_HIP],
            'left_hip': [self.mp_pose.PoseLandmark.LEFT_HIP],
            'right_knee': [self.mp_pose.PoseLandmark.RIGHT_KNEE],
            'left_knee': [self.mp_pose.PoseLandmark.LEFT_KNEE],
            'right_ankle': [self.mp_pose.PoseLandmark.RIGHT_ANKLE],
            'left_ankle': [self.mp_pose.PoseLandmark.LEFT_ANKLE]
        }
        
        # Store swing paths
        self.swing_paths = []
        
        # Number of points to normalize to
        self.normalized_points = 50

    def normalize_path(self, path):
        """Normalize a path to have a fixed number of points."""
        if len(path) < 2:
            return None
            
        # Create a parameter along the path
        t = np.linspace(0, 1, len(path))
        
        # Create interpolation functions for x, y, and z
        fx = interp1d(t, path[:, 0])
        fy = interp1d(t, path[:, 1])
        fz = interp1d(t, path[:, 2])
        
        # Create new parameter values
        t_new = np.linspace(0, 1, self.normalized_points)
        
        # Get interpolated values
        return np.column_stack((fx(t_new), fy(t_new), fz(t_new)))

    def process_video(self, video_path):
        """Process a video file and extract swing path."""
        print(f"\nProcessing: {video_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Scale down if resolution is too high
        scale_factor = 1.0
        max_dimension = 1920  # Maximum width or height
        if width > max_dimension or height > max_dimension:
            scale_factor = max_dimension / max(width, height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)
        
        print(f"Video properties: {width}x{height} @ {fps}fps")
        
        # Setup video writer
        output_path = os.path.join('output', f"tracked_{os.path.basename(video_path)}")
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Store swing path for this video
        swing_path = {name: [] for name in self.key_points.keys()}
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Scale down frame if necessary
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (width, height))
            
            # Process frame with pose detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # Store key points
                landmarks = results.pose_landmarks.landmark
                for point_name, landmark_list in self.key_points.items():
                    landmark = landmarks[landmark_list[0].value]
                    if landmark.visibility > 0.5:
                        swing_path[point_name].append([
                            landmark.x * width,
                            landmark.y * height,
                            landmark.z * width  # Using width as depth scale
                        ])
                    else:
                        swing_path[point_name].append([
                            float('nan'),
                            float('nan'),
                            float('nan')
                        ])
            
            # Write frame
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames")
        
        # Convert paths to numpy arrays and normalize
        valid_path = True
        for point_name in swing_path:
            if swing_path[point_name]:  # If we have data points
                path = np.array(swing_path[point_name])
                print(f"Points collected for {point_name}: {len(path)}")
                normalized = self.normalize_path(path)
                if normalized is not None:
                    swing_path[point_name] = normalized
                    print(f"Normalized points for {point_name}: {len(normalized)}")
                else:
                    valid_path = False
                    break
            else:
                valid_path = False
                break
        
        if valid_path:
            self.swing_paths.append(swing_path)
            print("Successfully added swing path to collection")
        else:
            print("Invalid swing path, skipping")
        
        # Release everything
        out.release()
        cap.release()
        print(f"\nSaved tracked video to: {output_path}")
        return True

    def create_animated_skeleton(self):
        """Create an animated 3D skeleton visualization."""
        if not self.swing_paths:
            print("No swing paths collected!")
            return
            
        print(f"Number of swing paths collected: {len(self.swing_paths)}")
        
        # Calculate average paths
        avg_paths = {}
        for point_name in self.key_points.keys():
            paths = [swing[point_name] for swing in self.swing_paths]
            avg_paths[point_name] = np.mean(paths, axis=0)
            print(f"Average path length for {point_name}: {len(avg_paths[point_name])}")
        
        # Create figure with white background
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        
        # Set up the skeleton connections
        connections = [
            # Head and upper body
            ('nose', 'right_shoulder'),
            ('nose', 'left_shoulder'),
            ('right_shoulder', 'left_shoulder'),
            
            # Arms
            ('right_shoulder', 'right_elbow'),
            ('left_shoulder', 'left_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_elbow', 'left_wrist'),
            
            # Torso
            ('right_shoulder', 'right_hip'),
            ('left_shoulder', 'left_hip'),
            ('right_hip', 'left_hip'),
            
            # Legs
            ('right_hip', 'right_knee'),
            ('left_hip', 'left_knee'),
            ('right_knee', 'right_ankle'),
            ('left_knee', 'left_ankle')
        ]
        
        # Initialize lines for animation
        lines = []
        for start_point, end_point in connections:
            line, = ax.plot([], [], [], 'k-', linewidth=2)
            lines.append(line)
        
        # Set consistent view limits
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_zlim([-200, 200])
        
        # Set view angle to show tennis player from side
        ax.view_init(elev=0, azim=-90)
        
        # Remove axes for cleaner visualization
        ax.set_axis_off()
        
        # Create animation
        num_frames = len(next(iter(avg_paths.values())))
        print(f"Creating animation with {num_frames} frames")
        
        def update(frame):
            ax.cla()
            ax.set_facecolor('white')
            
            # Draw connections
            for start_point, end_point in connections:
                if start_point in avg_paths and end_point in avg_paths:
                    start = avg_paths[start_point][frame]
                    end = avg_paths[end_point][frame]
                    ax.plot([start[0], end[0]], 
                           [start[1], end[1]], 
                           [start[2], end[2]], 'k-', linewidth=2)
            
            # Draw tennis racket
            if 'right_wrist' in avg_paths:
                wrist = avg_paths['right_wrist'][frame]
                elbow = avg_paths['right_elbow'][frame]
                arm_direction = wrist - elbow
                arm_direction = arm_direction / np.linalg.norm(arm_direction)
                
                racket_length = 100
                racket_head = wrist + arm_direction * racket_length
                
                # Draw racket handle
                ax.plot([wrist[0], racket_head[0]], 
                       [wrist[1], racket_head[1]], 
                       [wrist[2], racket_head[2]], 'r-', linewidth=3)
            
            # Set consistent view limits
            ax.set_xlim([-200, 200])
            ax.set_ylim([-200, 200])
            ax.set_zlim([-200, 200])
            
            # Set view angle to show tennis player from side
            ax.view_init(elev=0, azim=-90)
            
            # Remove axes for cleaner visualization
            ax.set_axis_off()
            
            # Add frame counter
            ax.text2D(0.02, 0.95, f'Frame: {frame}', transform=ax.transAxes)
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                     interval=50, blit=False)
        
        # Save animation
        anim.save('output/animated_skeleton.gif', writer='pillow')
        print("\nSaved animated skeleton to: output/animated_skeleton.gif")
        
        # Show the animation in an interactive window
        plt.show(block=True)

    def calculate_average_path(self):
        """Calculate and visualize the average swing path."""
        if not self.swing_paths:
            print("No swing paths collected!")
            return
            
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Plot each swing path
        colors = ['r', 'g', 'b']
        for point_name, color in zip(self.key_points.keys(), colors):
            # Plot individual paths
            for i, swing in enumerate(self.swing_paths):
                path = swing[point_name]
                plt.plot(path[:, 0], path[:, 1], f'{color}:', alpha=0.3, label=f'Swing {i+1}' if point_name == 'wrist' else '')
            
            # Calculate and plot average path
            paths = [swing[point_name] for swing in self.swing_paths]
            avg_path = np.mean(paths, axis=0)
            plt.plot(avg_path[:, 0], avg_path[:, 1], color, linewidth=2, label=f'Average {point_name}')
        
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.title("Federer's Forehand Swing Path Analysis")
        plt.xlabel("X position (pixels)")
        plt.ylabel("Y position (pixels)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('output/average_swing_path.png')
        print("\nSaved average swing path visualization to: output/average_swing_path.png")

def main():
    # Initialize tracker
    tracker = TennisTracker()
    
    # Process Serena Williams' forehand video
    video_path = "database/swilliams/forehand.mp4"
    if os.path.exists(video_path):
        tracker.process_video(video_path)
    else:
        print(f"Video not found: {video_path}")
    
    # Calculate and visualize average swing path
    tracker.calculate_average_path()
    
    # Create animated skeleton visualization
    tracker.create_animated_skeleton()

if __name__ == "__main__":
    main() 