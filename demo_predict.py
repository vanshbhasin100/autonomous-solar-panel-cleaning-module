import cv2
import sys
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# 1. Load trained best weights
MODEL_PATH = r"C:\Users\VANSH\Projects\solar-sweeper-ai\runs\solar_master_v8s\weights\best.pt"
model = YOLO(MODEL_PATH)

def run_demo():
    # File picker to select any image from your computer
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Solar Panel Image for Detection",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    
    if not file_path:
        print("No image selected.")
        return

    # 2. Run inference with custom confidence threshold
    results = model.predict(source=file_path, conf=0.20, imgsz=1024)
    
    # 3. Render and display the image with bounding boxes
    annotated_img = results[0].plot()
    
    cv2.imshow("Solar Sweeper AI - Debris & Dirt Detection Demo", annotated_img)
    print("\nPress any key or 'q' on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()