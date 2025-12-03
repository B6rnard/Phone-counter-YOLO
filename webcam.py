# webcam_yolo11.py
# Detect LEGO figures using a trained YOLOv11n model
# Press 'q' to quit
# pip install opencv-python
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from PIL import ImageGrab, Image, ImageTk
import threading
from PIL import ImageFont, ImageDraw

def main():
    # Select device
    device = 0 if torch.cuda.is_available() else "cpu"

    # Load trained YOLOv11 model
    # On windows, adjust path (most likely remove "6"
    model_path = "runs/detect/train6/weights/best.pt"
    model = YOLO(model_path).to(device)

    # Initialize webcam (fallback if multiple indices)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam found or cannot access camera.")

    print("Press 'q' to exit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Optional resize for faster inference
            # frame = cv2.resize(frame, (640, 480))

            # Run detection (use track() if object tracking is required)
            results = model(frame)

            # Count detected objects in this frame
            num_objects = len(results[0].boxes) if hasattr(results[0], "boxes") else 0

            # Print live counter (overwriting previous value)
            print(f"Objects detected: {num_objects}   ", end="\r", flush=True)

            # Optionally, write the count to a file for external display
            with open("count.txt", "w") as f:
                f.write(str(num_objects))

            # --- Kun vis tæller-displayet som overlay, ikke webcam-billedet ---
            # Lav et sort billede
            height, width = frame.shape[:2]
            overlay = 255 * np.ones((height, width, 3), dtype=np.uint8)

            # Tekst: overskrift
            title_text = "Antal telefoner fundet"
            font_title = cv2.FONT_HERSHEY_COMPLEX  # Bedre font til overskrift
            font_scale_title = 1.0  # Mindre tekst
            font_thickness_title = 2
            title_size, _ = cv2.getTextSize(title_text, font_title, font_scale_title, font_thickness_title)
            title_x = int((width - title_size[0]) / 2)
            title_y = int(height * 0.35)

            cv2.putText(
                overlay,
                title_text,
                (title_x, title_y),
                font_title,
                font_scale_title,
                (0, 0, 0),
                font_thickness_title,
                cv2.LINE_AA
            )

            # Tekst: mindre, mørkeblåt tal
            number_text = str(num_objects)
            font_number = cv2.FONT_HERSHEY_DUPLEX  # Bedre font til tal
            font_scale_number = 2.2  # Mindre tal
            font_thickness_number = 5
            number_size, _ = cv2.getTextSize(number_text, font_number, font_scale_number, font_thickness_number)
            number_x = int((width - number_size[0]) / 2)
            number_y = int(height * 0.65)

            cv2.putText(
                overlay,
                number_text,
                (number_x, number_y),
                font_number,
                font_scale_number,
                (139, 0, 64),  # Mørkeblå (BGR: 64, 0, 139) men OpenCV bruger BGR, så (139, 0, 64) er mørkeblå
                font_thickness_number,
                cv2.LINE_AA
            )

            # VIS OVERLAY-VINDUET
            cv2.imshow("Tæller Display", overlay)

            # Exit condition
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()