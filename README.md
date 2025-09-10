# 🚗 Autonomous Vehicle Lane Detection

A computer vision-based lane detection system using OpenCV for autonomous vehicles and ADAS applications.

## 🎯 Features

- **Real-time lane detection** from images, videos, or webcam
- **Robust edge detection** using Canny edge detection
- **Region of Interest (ROI)** masking for focused lane detection
- **Hough Line Transform** for accurate lane line detection
- **Multiple input modes**: Image, Video, Webcam
- **Clear visualization** with matplotlib plots

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Test

```bash
python test_lane_detection.py
```

This will:
- Create a synthetic road image
- Download a sample road image
- Run lane detection on both
- Display results with clear visualizations

### 3. Run Lane Detection

#### Image Mode (Default)
```bash
python lane_detection.py
```
Automatically downloads and processes a sample road image.

#### Custom Image
```bash
python lane_detection.py --mode image --input your_road_image.jpg
```

#### Video Mode
```bash
python lane_detection.py --mode video --input your_road_video.mp4 --output output_video.mp4
```

#### Real-time Webcam
```bash
python lane_detection.py --mode webcam
```
Press 'q' to quit.

## 📊 Output

The system provides clear visualizations showing:

1. **Original Image** - Input road image
2. **Canny Edges + ROI Mask** - Processed edges in region of interest
3. **Lane Lines Detected** - Final result with detected lane lines highlighted

## 🔧 Technical Details

### Pipeline Steps:
1. **Preprocessing**: Grayscale conversion and Gaussian blur
2. **Edge Detection**: Canny edge detection (50-150 thresholds)
3. **ROI Masking**: Focus on lower 60% of image (road area)
4. **Line Detection**: Hough Line Transform
5. **Visualization**: Overlay detected lines on original image

### Parameters:
- Canny thresholds: 50-150
- Hough parameters: ρ=1, θ=π/180, threshold=50
- Min line length: 20 pixels
- Max line gap: 300 pixels

## 📁 Project Structure

```
autonomous-vehicle-lane-detection/
├── lane_detection.py          # Main detection system
├── test_lane_detection.py     # Quick test script
├── requirements.txt           # Dependencies
├── README.md                 # This file
├── road_image.jpg           # Downloaded sample image
├── synthetic_road.jpg       # Generated test image
└── output_video.mp4         # Video output (if generated)
```

## 🎮 Usage Examples

### Basic Image Detection
```python
from lane_detection import LaneDetector

detector = LaneDetector()
img_rgb, masked_edges, combo_rgb, lines = detector.process_image('road_image.jpg')
detector.visualize_results(img_rgb, masked_edges, combo_rgb, lines)
```

### Video Processing
```python
from lane_detection import process_video

process_video('input_video.mp4', 'output_video.mp4')
```

## 🔍 Troubleshooting

### Common Issues:

1. **No lines detected**: Try adjusting Canny thresholds or Hough parameters
2. **Too many false lines**: Increase Hough threshold or min line length
3. **Missing lane lines**: Decrease Canny high threshold or Hough threshold
4. **Webcam not working**: Check camera permissions and availability

### Parameter Tuning:

```python
detector = LaneDetector()
detector.canny_low = 30      # Lower Canny threshold
detector.canny_high = 100    # Higher Canny threshold  
detector.hough_threshold = 30 # Lower Hough threshold
```

## 🚀 Advanced Features

- **Real-time processing** with webcam input
- **Video batch processing** with progress tracking
- **Configurable parameters** for different road conditions
- **Clean object-oriented design** for easy extension

## 📈 Performance

- **Image processing**: ~50-100ms per frame
- **Video processing**: Real-time capable on modern hardware
- **Memory efficient**: Minimal memory footprint
- **Cross-platform**: Works on Windows, macOS, Linux

## 🤝 Contributing

Feel free to enhance the system with:
- Deep learning integration (LaneNet, SCNN)
- Multi-lane detection
- Lane departure warning
- Curve detection
- Weather condition adaptation

## 📄 License

This project is open source and available under the MIT License.

---

**Ready to detect lanes? Run `python test_lane_detection.py` to get started!** 🚗✨
