from ultralytics import YOLO
import cv2
import time
import robomaster

from queue import Empty
from robomaster import robot
from robomaster import camera
import numpy as np
import os

IMAGE_CENTER_X = 320
REAL_LEGO_HEIGHT_M = 0.138
FOCAL_LENGTH = 314.0 

print('model')
model = YOLO("best.pt")


def search_stack_a(ep_robot, ep_camera, ep_chassis):
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)
        ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5)
        
        # get the yolo detection from frame
        if img is not None:
            start = time.time()
            if model.predictor:
                model.predictor.args.verbose = False
            result = model.predict(source=img, show=False)[0]

            #DIY visualization is much faster than show=True for some reason
            boxes = result.boxes
            distance_z = 0
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cv2.rectangle(img,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
                
                if box.conf[0] > 0.7:
                    print("Found stack A, returning")
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                    return

def align_a(ep_robot, ep_camera, ep_chassis):
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)
        ep_chassis.drive_speed(x=0, y=0, z=1, timeout=5)
        
        # get the yolo detection from frame
        if img is not None:
            start = time.time()
            if model.predictor:
                model.predictor.args.verbose = False
            result = model.predict(source=img, show=False)[0]

            #DIY visualization is much faster than show=True for some reason
            boxes = result.boxes
            distance_z = 0
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cv2.rectangle(img,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
                
                xyxy = box.xyxy.cpu().numpy().flatten()
                x1, y1, x2, y2 = xyxy
                box_center_x = (x1 + x2) / 2
                box_height_pixels = y2 - y1

                
                #angle_to_block = np.arccos(box_center_x - IMAGE_CENTER_X)/distance_z
                print("box_center", box_center_x)
                print("difference", abs(box_center_x - IMAGE_CENTER_X))
                if abs(box_center_x - IMAGE_CENTER_X) < 20:
                    print("Aligned to block A, returning")
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                    return

def approach_a(ep_robot, ep_camera, ep_chassis):
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)
        ep_chassis.drive_speed(x=0, y=0, z=1, timeout=5)
        
        # get the yolo detection from frame
        if img is not None:
            start = time.time()
            if model.predictor:
                model.predictor.args.verbose = False
            result = model.predict(source=img, show=False)[0]

            #DIY visualization is much faster than show=True for some reason
            boxes = result.boxes
            distance_z = 0
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cv2.rectangle(img,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
                
                xyxy = box.xyxy.cpu().numpy().flatten()
                x1, y1, x2, y2 = xyxy
                box_center_x = (x1 + x2) / 2
                box_height_pixels = y2 - y1
                distance_z = REAL_LEGO_HEIGHT_M / (box_height_pixels / FOCAL_LENGTH)
            
        
                while distance_z >= 0.27:
                    try:
                        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                    except Empty:
                        time.sleep(0.001)
                        continue
                    ep_chassis.drive_speed(x=0.1, y=0, z=0, timeout=5)
                return
                
if __name__ == '__main__':
    robomaster.config.ROBOT_IP_STR="192.168.50.114"
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")#(conn_type="sta", sn="3JKCH7T00100J0")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    # search_stack_a(ep_robot, ep_camera, ep_chassis)
    # align_a(ep_robot, ep_camera, ep_chassis)
    approach_a(ep_robot, ep_camera, ep_chassis)
    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel

    
