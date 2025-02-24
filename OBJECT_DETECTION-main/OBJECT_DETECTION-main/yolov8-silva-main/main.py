import random
import cv2
import torch
import speech_recognition as sr
from ultralytics import YOLO
import pyttsx3
from flask import Flask, render_template, Response

app = Flask(__name__)

# Opening the file in read mode
my_file = open("utils/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
my_file.close()

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt")

cap = cv2.VideoCapture(0)  # Open the default camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    global label
    started = False

    while True:
        command = "hello"

        if command == "hello":
            if not started:
                speak("Object detection is starting. Please wait.")
                started = True
                speak("Object detection has started.")
            else:
                speak("Object detection is already started.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                # Predict on image
                results = model.predict(frame)

                # Initialize dictionaries to track counts by position
                position_counts = {"left": {}, "center": {}, "right": {}}

                frame_width = frame.shape[1]  # Get frame width for position detection

                # Process detection results
                for result in results:
                    for box in result.boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = box
                        if conf > 0.25:
                            label = class_list[int(cls)]

                            # Determine object position
                            object_center_x = (x1 + x2) / 2
                            if object_center_x < frame_width / 3:
                                position = "left"
                            elif object_center_x > 2 * frame_width / 3:
                                position = "right"
                            else:
                                position = "center"

                            # Update counts for the position
                            if label in position_counts[position]:
                                position_counts[position][label] += 1
                            else:
                                position_counts[position][label] = 1

                            # Draw bounding box and label
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f} {position}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Speak detected objects and their counts by position
                detected_objects = []
                for position, objects in position_counts.items():
                    for label, count in objects.items():
                        detected_objects.append(f"{count} {label} on your {position}")

                if detected_objects:
                    detected_objects_str = ", ".join(detected_objects)
                    print("Detected Objects:", detected_objects_str)
                    speak("I see " + detected_objects_str)

                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            speak("Sorry, I didn't understand the command.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
