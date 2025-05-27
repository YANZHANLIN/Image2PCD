import numpy as np
import open3d as o3d
import re
import time
import cv2
import os
from PIL import Image

def parse_color_string(color_str):
    """Parse color string like '(255,128,64)' to RGB values"""
    match = re.match(r'\((\d+),(\d+),(\d+)\)', color_str)
    if match:
        return np.array([int(match.group(1)), int(match.group(2)), int(match.group(3))]) / 255.0
    return np.array([0, 0, 0])

def load_point_cloud_data(filename):
    """Load point cloud data from text file"""
    print(f"Loading data from {filename}...")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().replace('#', '').strip()
    columns = header.split()
    
    z_cols = [i for i, col in enumerate(columns) if col.startswith('Z')]
    color_cols = [i for i, col in enumerate(columns) if col.startswith('Color')]
    
    num_frames = len(z_cols)
    print(f"Found {num_frames} frames of data")
    
    data = []
    for line in lines[1:]:
        if line.strip():
            data.append(line.strip().split())
    
    data = np.array(data)
    
    x_coords = data[:, 0].astype(float)
    y_coords = data[:, 1].astype(float)
    
    z_frames = []
    for col_idx in z_cols:
        z_frames.append(data[:, col_idx].astype(float))
    
    color_frames = []
    for col_idx in color_cols:
        colors = np.array([parse_color_string(color_str) for color_str in data[:, col_idx]])
        color_frames.append(colors)
    
    return x_coords, y_coords, z_frames, color_frames, num_frames

def create_gpu_point_cloud_animation(x, y, z_frames, color_frames, output_filename='gpu_animation.mp4', 
                                   sample_rate=None, fps=30, rotation_degrees=360):
    """Create GPU-accelerated point cloud animation using Open3D with camera rotation
    
    Parameters:
    -----------
    rotation_degrees : int
        Total degrees to rotate the camera (default: 360 for full rotation)
    """
    
    # Auto-sample for performance if not specified
    if sample_rate is None:
        sample_rate = max(1, len(x) // 200000)  # Up to 200k points for GPU
    
    indices = np.arange(0, len(x), sample_rate)
    x_sample = x[indices]
    y_sample = y[indices]
    z_sample = [z[indices] for z in z_frames]
    color_sample = [colors[indices] for colors in color_frames]
    
    print(f"Using {len(x_sample)} points for GPU animation")
    
    # Create Open3D visualizer with GPU support
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, visible=False)  # High resolution
    except Exception as e:
        print(f"Warning: Could not create GPU visualizer: {e}")
        print("Falling back to CPU rendering...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=720, visible=False)
    
    # Enable GPU rendering if available
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # Black background
    render_option.point_size = 2.0
    
    # Create initial point cloud
    pcd = o3d.geometry.PointCloud()
    points_3d = np.column_stack((x_sample, y_sample, z_sample[0]))
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(color_sample[0])
    
    # Calculate the center of the point cloud
    center = pcd.get_center()
    print(f"Point cloud center: {center}")
    
    vis.add_geometry(pcd)
    
    # Set camera parameters
    ctr = vis.get_view_control()
    
    # Calculate camera distance based on point cloud bounds
    bounds = pcd.get_axis_aligned_bounding_box()
    max_bound = bounds.get_max_bound()
    min_bound = bounds.get_min_bound()
    size = max_bound - min_bound
    max_size = np.max(size)
    camera_distance = max_size * 2.0  # Adjust this multiplier to change zoom level
    
    # Set custom camera view from saved parameters
    # Create camera parameters from your saved extrinsic matrix
    camera_params = o3d.camera.PinholeCameraParameters()
    
    # Set intrinsic parameters
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1920, height=1080,
        fx=918.85295342, fy=918.85295342,
        cx=959.5, cy=530.0
    )
    
    # Set extrinsic parameters from your saved camera position
    extrinsic = np.array([
        [-9.98656078e-01,  2.05147318e-03, -5.17863744e-02,  6.71704549e+02],
        [ 9.61813989e-03,  9.89194765e-01, -1.46291518e-01, -7.97381191e+02],
        [ 5.09266974e-02, -1.46593002e-01, -9.87885096e-01,  1.86881693e+03],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])
    camera_params.extrinsic = extrinsic
    
    # Apply the saved camera parameters
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    
    print("Rendering frames...")
    frames = []
    
    # Calculate rotation steps
    num_frames = len(z_sample)
    rotation_per_frame = rotation_degrees / num_frames if num_frames > 0 else 0
    print(f"Number of frames: {num_frames}")
    print(f"Rotation per frame: {rotation_per_frame} degrees")
    total_rotation = 0
    
    for frame_idx in range(len(z_sample)):
        print(f"Rendering frame {frame_idx + 1}/{len(z_sample)}")
        
        # Update point cloud data
        points_3d = np.column_stack((x_sample, y_sample, z_sample[frame_idx]))
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(color_sample[frame_idx])
        
        # Rotate camera around the center
        if rotation_per_frame != 0:
            ctr.rotate(rotation_per_frame, 0)  # Rotate around vertical axis
            total_rotation += rotation_per_frame
            print(f"Total rotation so far: {total_rotation:.1f} degrees")
        
        # Update geometry
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image = vis.capture_screen_float_buffer(False)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        frames.append(image_np)
    
    if rotation_degrees > 0:
        print(f"Final total rotation: {total_rotation:.1f} degrees")
    
    vis.destroy_window()
    
    # Create video from frames
    print(f"Creating video {output_filename}...")
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"GPU animation saved as {output_filename}")

def create_interactive_gpu_viewer(x, y, z_frames, color_frames, sample_rate=None):
    """Create interactive GPU-accelerated point cloud viewer"""
    
    if sample_rate is None:
        sample_rate = max(1, len(x) // 200000)
    
    indices = np.arange(0, len(x), sample_rate)
    x_sample = x[indices]
    y_sample = y[indices]
    z_sample = [z[indices] for z in z_frames]
    color_sample = [colors[indices] for colors in color_frames]
    
    print(f"Using {len(x_sample)} points for interactive viewer")
    print("Controls:")
    print("- Mouse: Rotate view")
    print("- Mouse wheel: Zoom")
    print("- Arrow keys: Navigate frames")
    print("- S key: Save current camera parameters")
    print("- ESC: Exit")
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1920, height=1080)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    render_option.point_size = 2.0
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    points_3d = np.column_stack((x_sample, y_sample, z_sample[0]))
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(color_sample[0])
    
    vis.add_geometry(pcd)
    
    # Frame navigation
    current_frame = [0]  # Use list to modify in callback
    
    def next_frame(vis):
        if current_frame[0] < len(z_sample) - 1:
            current_frame[0] += 1
            update_frame()
        return False
    
    def prev_frame(vis):
        if current_frame[0] > 0:
            current_frame[0] -= 1
            update_frame()
        return False
    
    def save_camera_params(vis):
        """Save current camera parameters when 'S' is pressed"""
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        
        # Get camera position and orientation
        extrinsic = camera_params.extrinsic
        intrinsic = camera_params.intrinsic
        
        print("\n" + "="*50)
        print("CAMERA PARAMETERS SAVED!")
        print("="*50)
        print(f"Camera intrinsic matrix:")
        print(intrinsic.intrinsic_matrix)
        print(f"\nCamera extrinsic matrix:")
        print(extrinsic)
        
        # Extract more readable parameters
        import scipy.spatial.transform as R
        rotation_matrix = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        
        # Convert rotation matrix to Euler angles
        rotation = R.Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        
        print(f"\nReadable camera parameters:")
        print(f"Position (translation): {translation}")
        print(f"Rotation (Euler XYZ degrees): {euler_angles}")
        
        # Save to file for easy copying
        with open('camera_params.txt', 'w') as f:
            f.write("# Camera Parameters\n")
            f.write(f"# Position: {translation}\n")
            f.write(f"# Rotation (Euler XYZ degrees): {euler_angles}\n")
            f.write(f"# Extrinsic matrix:\n")
            f.write(f"{extrinsic}\n")
            f.write(f"# Intrinsic matrix:\n")
            f.write(f"{intrinsic.intrinsic_matrix}\n")
        
        print(f"Camera parameters saved to 'camera_params.txt'")
        print("="*50)
        return False
    
    def update_frame():
        frame_idx = current_frame[0]
        points_3d = np.column_stack((x_sample, y_sample, z_sample[frame_idx]))
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(color_sample[frame_idx])
        vis.update_geometry(pcd)
        print(f"Frame: {frame_idx + 1}/{len(z_sample)}")
    
    # Register key callbacks
    vis.register_key_callback(262, next_frame)  # Right arrow
    vis.register_key_callback(263, prev_frame)  # Left arrow
    vis.register_key_callback(83, save_camera_params)   # 'S' key
    
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    data_file = 'result/combined_xyz_positions.txt'
    
    try:
        x, y, z_frames, color_frames, num_frames = load_point_cloud_data(data_file)
        print(f"Loaded {len(x)} points across {num_frames} frames")
        
        # Create GPU-accelerated animation with rotation
        create_gpu_point_cloud_animation(x, y, z_frames, color_frames, 
                                       'gpu_point_cloud_animation.mp4', fps=5, rotation_degrees=0)
        
        # Optional: Create interactive viewer
        print("\nStarting interactive viewer...")
        create_interactive_gpu_viewer(x, y, z_frames, color_frames)
        
    except FileNotFoundError:
        print(f"Error: Could not find data file '{data_file}'")
    except Exception as e:
        print(f"Error: {e}")