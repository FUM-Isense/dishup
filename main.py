import time
import cv2
import espeakng
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
# import pyttsx3


engine = espeakng.Speaker(language='en',wpm=150)
# engine.setProperty('rate', 120)
last_time_say_lost = 0

def Say(text, reset_time=False):
    global last_time_say_lost

    if reset_time:
        last_time_say_lost = 0

    if time.time() - last_time_say_lost >= 2:
        last_time_say_lost = time.time()
        engine.say(f"  {text}  ")
        engine.wait()

def add_padding(bound, padding):
    x1, y1, x2, y2 = bound
    x1_padded = x1 - padding
    y1_padded = y1 - padding
    x2_padded = x2 + padding
    y2_padded = y2 + padding
    return x1_padded, y1_padded, x2_padded, y2_padded

Say("Start")

pipeline = rs.pipeline()
config = rs.config()
config.enable_device("246422071801")
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
pipeline_profile = pipeline.start(config)

frame_count = 0
tracker = cv2.TrackerCSRT_create()
tracking_initialized = False
detection_interval = 15
model = YOLO("yolo11n.pt")

last_seen_cup_time = 0 
glass_lost = False     
glass_ok_announced = False 
first_detection = False 
check_interval = 2 

while True:
    frame_count += 1
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    frame_rgb = np.asanyarray(color_frame.get_data())
    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

    current_time = time.time()

    # Perform detection every 2 seconds or when tracking is not initialized
    if not tracking_initialized:
        results = model(frame_rgb, verbose=False)
        cup_detected = False

        for result in results[0].boxes:
            if model.names[int(result.cls)] == 'cup' and result.conf > 0.5:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                w, h = x2 - x1, y2 - y1
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame_rgb, (x1, y1, w, h))
                tracking_initialized = True

                # Update last seen cup time
                last_seen_cup_time = current_time
                cup_detected = True

                # If it's the first detection or the cup was previously lost
                if not first_detection or glass_lost:
                    first_detection = True
                    glass_ok_announced = True
                    glass_lost = False
                    #Say("Glass is OK", True)

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_rgb, f"Cup Detected ({result.conf})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # If the glass is not detected and more than 2 seconds have passed since it was last seen
        if first_detection and not cup_detected:
            if current_time - last_seen_cup_time >= 2:
                if not glass_lost:
                    glass_lost = True
                    last_time_say_lost = 0  # Reset timer for announcing "lost"
                #Say("Glass is Lost")

    if tracking_initialized:
        success, box = tracker.update(frame_rgb)

        if success:
            # Successful tracking
            x, y, w, h = [int(v) for v in box]
            cup_bound = [x, y, x + w, y + h]
            x1, y1, x2, y2_cup = add_padding(cup_bound, -3)

            if frame_count % detection_interval == 0:
                tracking_initialized = False

            cup_roi = frame_rgb[y1:y2_cup, x1:x2]
            if cup_roi.size == 0:
                continue

            hsv_image = cv2.cvtColor(cup_roi, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 110, 45])
            upper_red = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2_cup), (0, 0, 255), 2)

            #cv2.imshow("RedMask", red_mask)
            white_pixels = np.sum(red_mask == 255)
            total_pixels = red_mask.size
            ratio = white_pixels / total_pixels

            print(f"Ratio : {ratio:.2f}", end='\r')

            if 0.5 < ratio <= 0.65:
                Say("Near")
            elif ratio > 0.65:
                Say("Full", reset_time=True)
        else:
            # Tracking was not successful
            if first_detection and current_time - last_seen_cup_time >= 2:
                if not glass_lost:
                    glass_lost = True
                    last_time_say_lost = 0  # Reset timer for announcing "lost"
                #Say("Glass is Lost")

    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('Frame', frame_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
