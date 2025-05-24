import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Function to extract XYZ positions from depth map
def extract_xyz_positions(depth_map, raw_image, mask=None):
    """
    Extracts XYZ positions from depth map.
    
    Parameters:
    - depth_map (np.ndarray): 2D array of depth values with shape (height, width).
    - raw_image (np.ndarray): 3D array of RGB values with shape (height, width, 3).
    - mask (np.ndarray, optional): 2D boolean array to filter points with shape (height, width).
    
    Returns:
    - xyz_positions (np.ndarray): Nx3 array of XYZ coordinates.
    - colors (np.ndarray): Nx3 array of RGB colors [0-255].
    """
    # Get the height and width from the depth map
    height, width = depth_map.shape
    
    # Generate grid of coordinates (v, u)
    v_coords, u_coords = np.indices((height, width))
    
    # If a mask is provided, apply it to filter valid points
    if mask is not None:
        valid = mask > 0
    else:
        valid = np.ones_like(depth_map, dtype=bool)
    
    # Extract the valid coordinates and depth values
    v_valid = v_coords[valid]
    u_valid = u_coords[valid]
    depth_valid = depth_map[valid]
    
    # Stack the coordinates to form XYZ positions (u=X, v=Y, depth=Z)
    xyz_positions = np.stack((u_valid, v_valid, depth_valid), axis=-1)
    
    # Extract the corresponding colors (keep as 0-255 range)
    colors = raw_image[valid]
    
    return xyz_positions, colors

def process_images_memory_efficient(image_files, output_file='result/combined_xyz_positions.txt', max_points=500000):
    """Process multiple images and save to a single file with memory management"""
    print(f"Processing {len(image_files)} images...")
    
    # Load model once
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    depth_anything = DepthAnythingV2(**model_configs["vitl"])
    depth_anything.load_state_dict(torch.load(
        "checkpoint/depth_anything_v2_vitl.pth",
        map_location=DEVICE,
    ))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Process first image to get dimensions and sampling
    raw = cv2.imread(image_files[0])
    depth = depth_anything.infer_image(raw)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    
    # Get valid points from first image to establish coordinate system
    xyz_first, colors_first = extract_xyz_positions(depth, raw)
    
    # Sample points if too many
    total_points = len(xyz_first)
    if total_points > max_points:
        print(f"Sampling {total_points} points down to {max_points}...")
        sample_indices = np.random.choice(total_points, max_points, replace=False)
        sample_indices.sort()  # Keep order for consistency
    else:
        sample_indices = np.arange(total_points)
    
    # Use sampled coordinates (X, Y will be same for all images)
    x_coords = xyz_first[sample_indices, 0]
    y_coords = xyz_first[sample_indices, 1]
    
    print(f"Using {len(sample_indices)} points for all images")
    
    # Create header
    header = "X Y " + " ".join([f"Z{i+1}" for i in range(len(image_files))]) + " " + \
             " ".join([f"Color{i+1}" for i in range(len(image_files))])
    
    # Collect all Z values and colors
    all_z_values = []
    all_colors = []
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing image {idx+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        raw = cv2.imread(image_file)
        if raw is None:
            print(f"Warning: Could not load {image_file}")
            continue
        
        depth = depth_anything.infer_image(raw)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        
        # Extract positions and colors
        xyz_pos, colors = extract_xyz_positions(depth, raw)
        
        # Use the same sampling as first image
        z_values = xyz_pos[sample_indices, 2]
        sampled_colors = colors[sample_indices]
        
        all_z_values.append(z_values)
        all_colors.append(sampled_colors)
        
        # Save individual depth image
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        cv2.imwrite(f'result/{image_name}.jpg', depth_norm)
    
    # Write to file
    print(f"Saving combined data to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"# {header}\n")
        
        # Write data in chunks to manage memory
        chunk_size = 10000
        for i in range(0, len(x_coords), chunk_size):
            chunk_end = min(i + chunk_size, len(x_coords))
            
            for j in range(i, chunk_end):
                line = [str(int(x_coords[j])), str(int(y_coords[j]))]
                
                # Add Z values for all images
                line.extend([f"{z_vals[j]:.6f}" for z_vals in all_z_values])
                
                # Add colors for all images
                line.extend([f"({c[0]},{c[1]},{c[2]})" for c in [colors[j] for colors in all_colors]])
                
                f.write(" ".join(line) + "\n")
    
    print(f"Data saved to {output_file}")
    print(f"Processed {len(all_z_values)} images with {len(x_coords)} points each")

if __name__ == '__main__':
    # Get all image files from the images folder
    import glob
    image_files = glob.glob("images/*.jpg") + glob.glob("images/*.png") + glob.glob("images/*.jpeg")
    image_files = sorted(image_files)
    
    if not image_files:
        print("No images found in the images folder!")
        exit()
    
    print(f"Found {len(image_files)} images to process")
    
    # Create result directory if it doesn't exist
    os.makedirs('result', exist_ok=True)
    
    # Process all images with memory management
    process_images_memory_efficient(image_files, max_points=500000)