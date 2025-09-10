"""
Create a realistic sample road image for testing
"""

import cv2
import numpy as np

def create_realistic_road_image():
    """Create a more realistic road image with proper perspective"""
    # Create a blank image
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    
    # Add sky gradient (light blue to white)
    for y in range(200):
        intensity = int(135 + (255-135) * (y/200))
        img[y, :] = [intensity, 206, 235]
    
    # Add road surface (dark gray with some texture)
    road_color = [40, 40, 40]
    img[200:, :] = road_color
    
    # Add road texture (asphalt pattern)
    for y in range(200, 540, 3):
        for x in range(0, 960, 15):
            if (x + y) % 30 < 15:
                img[y:y+2, x:x+10] = [35, 35, 35]
    
    # Add lane markings with proper perspective
    # Left lane line (curved perspective)
    left_points = []
    for y in range(200, 540, 10):
        x = int(300 + (y-200) * 0.3 + np.sin((y-200) * 0.01) * 20)
        left_points.append((x, y))
    
    for i in range(len(left_points)-1):
        cv2.line(img, left_points[i], left_points[i+1], (255, 255, 255), 6)
    
    # Right lane line (curved perspective)
    right_points = []
    for y in range(200, 540, 10):
        x = int(650 + (y-200) * 0.2 + np.sin((y-200) * 0.008) * 15)
        right_points.append((x, y))
    
    for i in range(len(right_points)-1):
        cv2.line(img, right_points[i], right_points[i+1], (255, 255, 255), 6)
    
    # Add center dashed line
    for y in range(200, 540, 30):
        x1 = int(450 + (y-200) * 0.25)
        x2 = int(470 + (y-200) * 0.25)
        cv2.line(img, (x1, y), (x2, y+15), (255, 255, 255), 4)
    
    # Add some road wear and markings
    cv2.line(img, (100, 400), (200, 420), (60, 60, 60), 3)  # Road wear
    cv2.line(img, (750, 380), (850, 400), (60, 60, 60), 3)  # Road wear
    
    # Add horizon line
    cv2.line(img, (0, 200), (960, 200), (100, 100, 100), 2)
    
    return img

if __name__ == "__main__":
    img = create_realistic_road_image()
    cv2.imwrite('realistic_road.jpg', img)
    print("✓ Created realistic road image: realistic_road.jpg")
    
    # Display the image
    import matplotlib.pyplot as plt
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.title("Generated Realistic Road Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


