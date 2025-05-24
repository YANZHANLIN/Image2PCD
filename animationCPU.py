import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
import cv2
from mpl_toolkits.mplot3d import Axes3D

def parse_color_string(color_str):
    """Parse color string like '(255,128,64)' to RGB values"""
    match = re.match(r'\((\d+),(\d+),(\d+)\)', color_str)
    if match:
        return np.array([int(match.group(1)), int(match.group(2)), int(match.group(3))]) / 255.0
    return np.array([0, 0, 0])  # default to black if parsing fails

def load_point_cloud_data(filename):
    """Load point cloud data from text file"""
    print(f"Loading data from {filename}...")
    
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header to find number of Z and Color columns
    header = lines[0].strip().replace('#', '').strip()
    columns = header.split()
    
    # Find Z and Color column indices
    z_cols = [i for i, col in enumerate(columns) if col.startswith('Z')]
    color_cols = [i for i, col in enumerate(columns) if col.startswith('Color')]
    
    num_frames = len(z_cols)
    print(f"Found {num_frames} frames of data")
    
    # Load data (skip header)
    data = []
    for line in lines[1:]:
        if line.strip():
            data.append(line.strip().split())
    
    data = np.array(data)
    
    # Extract X, Y coordinates (same for all frames)
    x_coords = data[:, 0].astype(float)
    y_coords = data[:, 1].astype(float)
    
    # Extract Z values for each frame
    z_frames = []
    for col_idx in z_cols:
        z_frames.append(data[:, col_idx].astype(float))
    
    # Extract colors for each frame
    color_frames = []
    for col_idx in color_cols:
        colors = np.array([parse_color_string(color_str) for color_str in data[:, col_idx]])
        color_frames.append(colors)
    
    return x_coords, y_coords, z_frames, color_frames, num_frames

def create_3d_animation(x, y, z_frames, color_frames, output_filename='point_cloud_animation.mp4'):
    """Create 3D animation of point cloud changes"""
    
    # Sample points for better performance (adjust as needed)
    sample_rate = max(1, len(x) // 10000)  # Limit to ~10k points for performance
    indices = np.arange(0, len(x), sample_rate)
    
    x_sample = x[indices]
    y_sample = y[indices]
    z_sample = [z[indices] for z in z_frames]
    color_sample = [colors[indices] for colors in color_frames]
    
    print(f"Using {len(x_sample)} sampled points for animation")
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Animation - Z and Color Changes')
    
    # Calculate plot limits
    x_min, x_max = x_sample.min(), x_sample.max()
    y_min, y_max = y_sample.min(), y_sample.max()
    z_min = min([z.min() for z in z_sample])
    z_max = max([z.max() for z in z_sample])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Initialize scatter plot
    scat = ax.scatter(x_sample, y_sample, z_sample[0], 
                     c=color_sample[0], s=1, alpha=0.6)
    
    # Animation function
    def animate(frame):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud Animation - Frame {frame + 1}/{len(z_sample)}')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Update scatter plot with new Z values and colors
        ax.scatter(x_sample, y_sample, z_sample[frame], 
                  c=color_sample[frame], s=1, alpha=0.6)
        
        return ax,
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, animate, frames=len(z_sample), 
                        interval=500, blit=False, repeat=True)
    
    # Save animation
    print(f"Saving animation to {output_filename}...")
    Writer = cv2.VideoWriter_fourcc(*'mp4v')
    anim.save(output_filename, writer='ffmpeg', fps=2)
    
    print(f"Animation saved as {output_filename}")
    
    # Show the plot
    plt.show()

def create_2d_depth_animation(x, y, z_frames, output_filename='depth_animation.mp4'):
    """Create 2D depth map animation showing Z changes as heatmap"""
    
    # Get image dimensions
    height = int(y.max()) + 1
    width = int(x.max()) + 1
    
    print(f"Creating depth animation with dimensions: {width}x{height}")
    
    # Create depth images for each frame
    depth_images = []
    for z_frame in z_frames:
        depth_img = np.zeros((height, width))
        # Map points to image coordinates
        for i in range(len(x)):
            row, col = int(y[i]), int(x[i])
            if 0 <= row < height and 0 <= col < width:
                depth_img[row, col] = z_frame[i]
        depth_images.append(depth_img)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize depth values across all frames for consistent color scale
    z_min = min([z.min() for z in z_frames])
    z_max = max([z.max() for z in z_frames])
    
    im = ax.imshow(depth_images[0], cmap='viridis', vmin=z_min, vmax=z_max)
    ax.set_title('Depth Map Animation - Frame 1')
    plt.colorbar(im, ax=ax, label='Depth (Z)')
    
    def animate_depth(frame):
        im.set_array(depth_images[frame])
        ax.set_title(f'Depth Map Animation - Frame {frame + 1}/{len(depth_images)}')
        return [im]
    
    anim = FuncAnimation(fig, animate_depth, frames=len(depth_images), 
                        interval=500, blit=True, repeat=True)
    
    print(f"Saving depth animation to {output_filename}...")
    anim.save(output_filename, writer='ffmpeg', fps=2)
    print(f"Depth animation saved as {output_filename}")
    
    plt.show()

if __name__ == '__main__':
    # Load your data file
    data_file = 'result/combined_xyz_positions.txt'  # Update path if needed
    
    try:
        x, y, z_frames, color_frames, num_frames = load_point_cloud_data(data_file)
        
        print(f"Loaded {len(x)} points across {num_frames} frames")
        
        # Create 3D animation showing both Z and color changes
        create_3d_animation(x, y, z_frames, color_frames, 'point_cloud_3d_animation.mp4')
        
        # Create 2D depth animation showing Z changes as heatmap
        create_2d_depth_animation(x, y, z_frames, 'depth_map_animation.mp4')
        
    except FileNotFoundError:
        print(f"Error: Could not find data file '{data_file}'")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"Error processing data: {e}")