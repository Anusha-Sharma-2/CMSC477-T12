from ultralytics import YOLO
import cv2
import time
import robomaster

from queue import Empty
from robomaster import robot
from robomaster import camera
import numpy as np
import os

print('model')
# model = YOLO("yolo11n.pt")

# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)
    
robomaster.config.ROBOT_IP_STR="192.168.50.114"
ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta")
np.set_printoptions(precision=3, suppress=True, linewidth=120)
ep_chassis = ep_robot.chassis
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

output_dir = 'block_images'
if not os.path.exists(output_dir):
    print("made folder")
    os.makedirs(output_dir)

frame_count = 0

while frame_count < 300:
    #ret, frame = vid.read()
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if frame is not None:
            start = time.time()
            # if model.predictor:
            #     model.predictor.args.verbose = False
            # result = model.predict(source=frame, show=False)[0]

            # DIY visualization is much faster than show=True for some reason
            # boxes = result.boxes
            # for box in boxes:
            #     xyxy = box.xyxy.cpu().numpy().flatten()
            #     cv2.rectangle(frame,
            #                 (int(xyxy[0]), int(xyxy[1])), 
            #                 (int(xyxy[2]), int(xyxy[3])),
            #                 color=(0, 0, 255), thickness=2)
                
            cv2.imshow('frame', frame)
            if frame_count % 10 == 0:
                print(f"Saving {frame_count}")
                frame_filename = os.path.join(output_dir, f'BLOCK7{frame_count}.jpg')
                cv2.imwrite(frame_filename, frame)
            frame_count += 1
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except Empty:
        time.sleep(0.001)
        continue
        
    
ep_camera.stop_video_stream()
ep_robot.close()