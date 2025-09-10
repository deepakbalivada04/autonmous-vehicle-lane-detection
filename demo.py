"""
Complete Demo of Autonomous Vehicle Lane Detection System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from lane_detection import LaneDetector
import os

def run_complete_demo():
    """Run a complete demonstration of the lane detection system"""
    print("🚗 AUTONOMOUS VEHICLE LANE DETECTION DEMO")
    print("=" * 60)
    
    detector = LaneDetector()
    
    # Demo 1: Synthetic Road Image
    print("\n📸 DEMO 1: Synthetic Road Image")
    print("-" * 40)
    
    if os.path.exists('synthetic_road.jpg'):
        try:
            img_rgb, masked_edges, combo_rgb, lines = detector.process_image('synthetic_road.jpg')
            detector.visualize_results(img_rgb, masked_edges, combo_rgb, lines)
            print("✓ Synthetic image processed successfully!")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("⚠ Synthetic image not found. Run test_lane_detection.py first.")
    
    # Demo 2: Realistic Road Image
    print("\n📸 DEMO 2: Realistic Road Image")
    print("-" * 40)
    
    if os.path.exists('realistic_road.jpg'):
        try:
            img_rgb, masked_edges, combo_rgb, lines = detector.process_image('realistic_road.jpg')
            detector.visualize_results(img_rgb, masked_edges, combo_rgb, lines)
            print("✓ Realistic image processed successfully!")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("⚠ Realistic image not found. Run create_sample_image.py first.")
    
    # Demo 3: Show Processing Steps
    print("\n🔍 DEMO 3: Processing Steps Breakdown")
    print("-" * 40)
    
    if os.path.exists('realistic_road.jpg'):
        img = cv2.imread('realistic_road.jpg')
        img = cv2.resize(img, (960, 540))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step by step processing
        gray, blur = detector.preprocess_image(img)
        edges = detector.detect_edges(blur)
        masked_edges = detector.create_roi_mask(edges)
        lines = detector.detect_lines(masked_edges)
        line_img = detector.draw_lines(img, lines)
        combo = cv2.addWeighted(img, 0.8, line_img, 1, 0)
        combo_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
        
        # Display all steps
        plt.figure(figsize=(20, 12))
        
        plt.subplot(2, 3, 1)
        plt.title("1. Original Image", fontsize=14, fontweight='bold')
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("2. Grayscale + Blur", fontsize=14, fontweight='bold')
        plt.imshow(blur, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("3. Canny Edge Detection", fontsize=14, fontweight='bold')
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("4. ROI Mask Applied", fontsize=14, fontweight='bold')
        plt.imshow(masked_edges, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("5. Detected Lines", fontsize=14, fontweight='bold')
        line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        plt.imshow(line_img_rgb)
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.title("6. Final Result", fontsize=14, fontweight='bold')
        plt.imshow(combo_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✓ Processing steps displayed! Detected {len(lines) if lines is not None else 0} lines")
    
    # Demo 4: Parameter Tuning Example
    print("\n⚙️ DEMO 4: Parameter Tuning Example")
    print("-" * 40)
    
    if os.path.exists('realistic_road.jpg'):
        # Original parameters
        img_rgb, masked_edges, combo_rgb, lines_orig = detector.process_image('realistic_road.jpg')
        
        # Tuned parameters
        detector.canny_low = 30
        detector.canny_high = 100
        detector.hough_threshold = 30
        img_rgb2, masked_edges2, combo_rgb2, lines_tuned = detector.process_image('realistic_road.jpg')
        
        # Reset parameters
        detector.canny_low = 50
        detector.canny_high = 150
        detector.hough_threshold = 50
        
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.title(f"Original Parameters\n({len(lines_orig) if lines_orig is not None else 0} lines)", 
                 fontsize=14, fontweight='bold')
        plt.imshow(combo_rgb)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Tuned Parameters\n({len(lines_tuned) if lines_tuned is not None else 0} lines)", 
                 fontsize=14, fontweight='bold')
        plt.imshow(combo_rgb2)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Parameter tuning comparison displayed!")
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED!")
    print("\n📋 Available Commands:")
    print("• python lane_detection.py                    # Default image mode")
    print("• python lane_detection.py --mode webcam      # Real-time webcam")
    print("• python lane_detection.py --mode video --input video.mp4  # Video processing")
    print("• python test_lane_detection.py               # Quick test")
    print("\n🚀 The system is ready for autonomous vehicle lane detection!")

if __name__ == "__main__":
    run_complete_demo()
