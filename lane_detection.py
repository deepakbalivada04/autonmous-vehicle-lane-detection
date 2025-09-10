"""
Autonomous Vehicle Lane Detection using Computer Vision

This project develops a lane detection system using OpenCV image processing techniques.
It processes video frames from road footage, detects lane markings, and highlights 
lane boundaries for autonomous driving or advanced driver-assistance systems (ADAS).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import argparse
from typing import Tuple, List, Optional

class LaneDetector:
    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 50
        self.hough_min_line_length = 20
        self.hough_max_line_gap = 300
        
    def download_sample_image(self) -> str:
        """Download a sample road image for testing"""
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Driving_on_a_clear_day_in_Wiesbaden_%28Germany%29_-_2012-09-21.jpg/640px-Driving_on_a_clear_day_in_Wiesbaden_%28Germany%29_-_2012-09-21.jpg'
        img_path = 'road_image.jpg'
        
        if not os.path.exists(img_path):
            print("Downloading sample road image...")
            try:
                urllib.request.urlretrieve(image_url, img_path)
                print("✓ Sample road image downloaded successfully!")
            except Exception as e:
                print(f"✗ Failed to download image: {e}")
                return None
        else:
            print("✓ Sample road image already exists!")
        
        return img_path
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing steps: grayscale and Gaussian blur"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, blur
    
    def detect_edges(self, blur: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection"""
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        return edges
    
    def create_roi_mask(self, edges: np.ndarray) -> np.ndarray:
        """Create region of interest mask to focus on lane area"""
        mask = np.zeros_like(edges)
        height, width = edges.shape
        
        # Define polygon for region of interest (lower half of image)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.6)),
            (0, int(height * 0.6))
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges
    
    def detect_lines(self, masked_edges: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines using Hough Line Transform"""
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        return lines
    
    def draw_lines(self, img: np.ndarray, lines: np.ndarray) -> np.ndarray:
        """Draw detected lines on the image"""
        line_img = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
        return line_img
    
    def process_image(self, img_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Complete lane detection pipeline for a single image"""
        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image from {img_path}")
        
        img = cv2.resize(img, (960, 540))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocessing
        gray, blur = self.preprocess_image(img)
        
        # Edge detection
        edges = self.detect_edges(blur)
        
        # Region of interest
        masked_edges = self.create_roi_mask(edges)
        
        # Line detection
        lines = self.detect_lines(masked_edges)
        
        # Draw lines
        line_img = self.draw_lines(img, lines)
        
        # Combine original with detected lines
        combo = cv2.addWeighted(img, 0.8, line_img, 1, 0)
        combo_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
        
        return img_rgb, masked_edges, combo_rgb, lines
    
    def visualize_results(self, img_rgb: np.ndarray, masked_edges: np.ndarray, 
                         combo_rgb: np.ndarray, lines: Optional[np.ndarray]):
        """Display the results in a matplotlib figure"""
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image", fontsize=14, fontweight='bold')
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Canny Edges + ROI Mask", fontsize=14, fontweight='bold')
        plt.imshow(masked_edges, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Lane Lines Detected ({len(lines) if lines is not None else 0} lines)", 
                 fontsize=14, fontweight='bold')
        plt.imshow(combo_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if lines is not None:
            print(f"✓ Detected {len(lines)} lane lines")
        else:
            print("⚠ No lane lines detected")

def process_video(video_path: str, output_path: str = 'output_video.mp4'):
    """Process video file for lane detection"""
    detector = LaneDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame
        frame = cv2.resize(frame, (960, 540))
        
        # Process frame
        gray, blur = detector.preprocess_image(frame)
        edges = detector.detect_edges(blur)
        masked_edges = detector.create_roi_mask(edges)
        lines = detector.detect_lines(masked_edges)
        line_img = detector.draw_lines(frame, lines)
        result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        
        # Resize back to original dimensions
        result = cv2.resize(result, (width, height))
        out.write(result)
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    out.release()
    print(f"✓ Video processing complete! Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Autonomous Vehicle Lane Detection')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], default='image',
                       help='Detection mode: image, video, or webcam')
    parser.add_argument('--input', type=str, help='Input image or video file path')
    parser.add_argument('--output', type=str, default='output_video.mp4',
                       help='Output video file path (for video mode)')
    
    args = parser.parse_args()
    
    detector = LaneDetector()
    
    if args.mode == 'image':
        if args.input:
            img_path = args.input
        else:
            img_path = detector.download_sample_image()
            if img_path is None:
                return
        
        print("🚗 Starting Lane Detection on Image...")
        try:
            img_rgb, masked_edges, combo_rgb, lines = detector.process_image(img_path)
            detector.visualize_results(img_rgb, masked_edges, combo_rgb, lines)
            print("✓ Lane detection completed successfully!")
        except Exception as e:
            print(f"✗ Error processing image: {e}")
    
    elif args.mode == 'video':
        if not args.input:
            print("✗ Please provide input video file with --input")
            return
        
        print("🚗 Starting Lane Detection on Video...")
        process_video(args.input, args.output)
    
    elif args.mode == 'webcam':
        print("🚗 Starting Real-time Lane Detection from Webcam...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            gray, blur = detector.preprocess_image(frame)
            edges = detector.detect_edges(blur)
            masked_edges = detector.create_roi_mask(edges)
            lines = detector.detect_lines(masked_edges)
            line_img = detector.draw_lines(frame, lines)
            result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
            
            # Display result
            cv2.imshow('Lane Detection', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam processing stopped")

if __name__ == "__main__":
    main()
