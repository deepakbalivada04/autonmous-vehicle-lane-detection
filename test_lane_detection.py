"""
Quick test script to run lane detection with sample data
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
from lane_detection import LaneDetector

def create_sample_road_image():
    """Create a simple synthetic road image for testing"""
    # Create a blank image
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    
    # Add road surface (dark gray)
    img[200:, :] = [50, 50, 50]
    
    # Add lane markings (white lines)
    # Left lane line
    cv2.line(img, (300, 540), (400, 200), (255, 255, 255), 8)
    # Right lane line  
    cv2.line(img, (600, 540), (700, 200), (255, 255, 255), 8)
    
    # Add some road texture
    for i in range(0, 960, 20):
        cv2.line(img, (i, 200), (i, 540), (40, 40, 40), 1)
    
    # Add sky (light blue)
    img[:200, :] = [135, 206, 235]
    
    return img

def download_test_video():
    """Download a sample road video for testing"""
    # This is a placeholder - in practice you'd use a real road video
    print("For video testing, please provide your own road video file.")
    print("You can use any MP4 video file with road footage.")
    return None

def run_quick_test():
    """Run a quick test of the lane detection system"""
    print("🚗 Running Quick Lane Detection Test...")
    print("=" * 50)
    
    detector = LaneDetector()
    
    # Test 1: Synthetic image
    print("\n1. Testing with synthetic road image...")
    synthetic_img = create_sample_road_image()
    cv2.imwrite('synthetic_road.jpg', synthetic_img)
    
    try:
        img_rgb, masked_edges, combo_rgb, lines = detector.process_image('synthetic_road.jpg')
        detector.visualize_results(img_rgb, masked_edges, combo_rgb, lines)
        print("✓ Synthetic image test completed!")
    except Exception as e:
        print(f"✗ Synthetic image test failed: {e}")
    
    # Test 2: Real road image
    print("\n2. Testing with real road image...")
    img_path = detector.download_sample_image()
    if img_path:
        try:
            img_rgb, masked_edges, combo_rgb, lines = detector.process_image(img_path)
            detector.visualize_results(img_rgb, masked_edges, combo_rgb, lines)
            print("✓ Real image test completed!")
        except Exception as e:
            print(f"✗ Real image test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick test completed!")
    print("\nTo run more tests:")
    print("• For image: python lane_detection.py --mode image --input your_image.jpg")
    print("• For video: python lane_detection.py --mode video --input your_video.mp4")
    print("• For webcam: python lane_detection.py --mode webcam")

if __name__ == "__main__":
    run_quick_test()


