# Image2PCD: Image to Point Cloud Converter and Visualizer

A powerful tool for converting images into 3D point clouds and creating stunning visualizations. This project uses depth estimation to transform 2D images into 3D point clouds and provides both automated animations and interactive viewing capabilities.

## Features

- Convert multiple images into 3D point clouds using depth estimation
- Memory-efficient processing with automatic point sampling
- GPU-accelerated visualization (with CPU fallback)
- Create animated videos of point clouds with customizable camera rotation
- Interactive 3D viewer for exploring point clouds
- Support for multiple frames/animations
- High-resolution output (1920x1080)

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Open3D
- PyTorch
- Depth-Anything-V2 model
- CUDA-capable GPU (optional, for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YANZHANLIN/Image2PCD.git
cd Image2PCD
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Depth-Anything-V2 model checkpoint and place it in the `checkpoint` directory.

## Project Structure

```
Image2PCD/
├── images/             # Input images directory
├── result/            # Output directory for results
├── checkpoint/        # Model weights directory
├── runmulti.py        # Main processing script
├── animationGPU2.py   # Animation and visualization script
├── requirements.txt   # Python dependencies
├── LICENSE           # MIT License
└── README.md         # This file
```

## Usage

### 1. Processing Images

Place your input images in the `images` folder. Supported formats: JPG, PNG, JPEG.

Run the processing script:
```bash
python runmulti.py
```

This will:
- Generate depth maps for each image
- Create point cloud data
- Save depth maps as JPG files in the `result` directory
- Create a combined point cloud data file (`combined_xyz_positions.txt`)

### 2. Creating Animations

Run the animation script:
```bash
python animationGPU2.py
```

This will:
- Create a video animation of the point cloud
- Launch an interactive 3D viewer

#### Animation Options

You can customize the animation by modifying parameters in `animationGPU2.py`:
- `fps`: Frames per second (default: 5)
- `rotation_degrees`: Camera rotation angle (default: 360, set to 0 for no rotation)
- `sample_rate`: Point sampling rate (automatically calculated for optimal performance)

### 3. Interactive Viewer

The interactive viewer provides the following controls:
- Mouse: Rotate view
- Mouse wheel: Zoom
- Arrow keys: Navigate between frames
- ESC: Exit

## Output

The project generates:
1. Depth maps in the `result` directory
2. A combined point cloud data file (`combined_xyz_positions.txt`)
3. An MP4 video animation (`gpu_point_cloud_animation.mp4`)

## Performance Considerations

- The system automatically samples points to maintain performance (default max: 500,000 points)
- GPU acceleration is used when available
- Memory-efficient processing with chunk-based file writing
- Automatic fallback to CPU rendering if GPU is unavailable

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[no license]

## Acknowledgments

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) model for depth estimation
- [Open3D](http://www.open3d.org/) for 3D visualization
- [OpenCV](https://opencv.org/) for image processing

