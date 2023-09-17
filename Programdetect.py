import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
import cv2


# Load YOLO model
model = YOLO("yolov8n.pt")

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes in frame"""
    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf

        # Draw bounding box
        annotator.box_label(box=coordinator, label=class_name, color=colors(class_id, True))

    return annotator.result()

def detect_motorcycle(frame):
    """Detect Motorcycle from input image"""
    consider_classes = [0, 3]
    confidence_threshold = 0.5
    results = model.predict(frame, conf=confidence_threshold, classes=consider_classes)

    for result in results:
        # Each frame draw box in frame
        frame = draw_boxes(frame, result.boxes)

    return frame

# Function to open camera and perform detection
def open_camera():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Perform motorcycle detection
            frame_with_boxes = detect_motorcycle(frame)

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
                # Perform motorcycle detection
                frame_with_boxes = detect_motorcycle(frame)

                # Display the frame with bounding boxes
                cv2.imshow("Video", frame_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

# Function to handle the "Start Detection" button click event
def start_detection():
    # Check if the user has selected either camera or video
    if var.get() == 1:
        open_camera()
    elif var.get() == 2:
        open_video()
    else:
        messagebox.showinfo("Error", "Please select Camera or Video.")

# Create the main GUI window
root = tk.Tk()
root.title("Motorcycle Detection GUI")
root.geometry("500x200")

# Create a label
label = tk.Label(root, text="Motorcycle Detection", font=("Arial", 20))
label.pack()

# Create radio buttons for camera and video options
var = tk.IntVar()
camera_radio = tk.Radiobutton(root, text="Camera", variable=var, value=1)
video_radio = tk.Radiobutton(root, text="Video (mp4)", variable=var, value=2)

# Create a button to start detection
start_button = tk.Button(root, text="Start Detection", command=start_detection)

# Pack the radio buttons and start button
camera_radio.pack()
video_radio.pack()
start_button.pack()

# Start the GUI main loop
root.mainloop()