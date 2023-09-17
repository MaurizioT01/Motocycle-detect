import tkinter as tk
from tkinter import filedialog
from ultralytics.models import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Function to open camera and perform detection
def open_camera():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Perform object detection
            frame_with_boxes = detect_objects(frame)

            # Display the frame with bounding boxes
            cv2.imshow("Camera", frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to open a video file and perform detection
def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if file_path:
        cap = cv2.VideoCapture(file_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Perform object detection
                frame_with_boxes = detect_objects(frame)

                # Display the frame with bounding boxes
                cv2.imshow("Video", frame_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

# Function to perform object detection on an image
def detect_objects(frame):
    consider_classes = None  # Detect all classes
    confidence_threshold = 0.5
    results = model.predict(frame, conf=confidence_threshold)

    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame

# Function to draw bounding boxes on an image
def draw_boxes(frame, boxes):
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf
        annotator.box_label(box=coordinator, label=class_name, color=colors(class_id, True))

    return annotator.result()

# Create the main GUI window with size 500*400
root = tk.Tk()
root.title("Motorcycle Detection GUI")
root.geometry("500x200")  # Set the size to 500*400

# Add a Label for the text "OBJECT DETECTION" with font size 20
label = tk.Label(root, text="MOTORCYCLLE DETECTION", font=("Arial", 20))
label.pack()

# Create buttons to open camera and video file
camera_button = tk.Button(root, text="Open Camera", command=open_camera, font=("Arial", 20))
video_button = tk.Button(root, text="Open Video", command=open_video, font=("Arial", 20))

# Pack the buttons
camera_button.pack()
video_button.pack()

# Start the GUI main loop
root.mainloop()