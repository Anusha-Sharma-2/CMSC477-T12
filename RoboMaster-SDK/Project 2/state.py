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

def openclaw (ep_robot, ep_camera, ep_chassis, ep_gripper):
    # Move the arm to the "retracted" position
    ep_robot.robotic_arm.moveto(x=100, y=30).wait_for_completed()
    # Move the arm forward and down in order to pickup an object
    # (we do this in two moves to avoid the "keep out zone" where the robot may hit itself)
    ep_robot.robotic_arm.moveto(x=180, y=30).wait_for_completed()
    ep_robot.robotic_arm.moveto(x=180, y=-50).wait_for_completed()
    
    # Open the gripper
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()
    
def closeclaw(ep_robot, ep_camera, ep_chassis, ep_gripper):
    # Close the gripper on the object
    ep_chassis.drive_speed(x=0, y=0, z=0)
    ep_gripper.close(power=50)
    time.sleep(1)
    ep_robot.robotic_arm.moveto(x=180, y=-50).wait_for_completed()
    ep_robot.robotic_arm.moveto(x=180, y=30).wait_for_completed()
    ep_robot.robotic_arm.moveto(x=100, y=30).wait_for_completed()
    
    ep_gripper.pause()
    
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
        ep_chassis.drive_speed(x=0, y=0, z=3, timeout=5)
        
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
                if abs(box_center_x - IMAGE_CENTER_X) < 5:
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
        ep_chassis.drive_speed(x=0.1, y=0, z=0, timeout=5)
        
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

                print("distance: ", distance_z)
                if distance_z <= 0.3:
                    try:
                        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                    except Empty:
                        time.sleep(0.001)
                        continue
                    print("distance: ", distance_z)
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    return

def carry_side(ep_robot, ep_camera, ep_chasses):
    # move backward 0.5
    ep_chassis.drive_speed(x=-0.2, y=0, z=0, timeout=5)
    time.sleep(5)
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                
if __name__ == '__main__':
    robomaster.config.ROBOT_IP_STR="192.168.50.114"
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")#(conn_type="sta", sn="3JKCH7T00100J0")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_gripper = ep_robot.gripper
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    search_stack_a(ep_robot, ep_camera, ep_chassis)
    align_a(ep_robot, ep_camera, ep_chassis)
    openclaw (ep_robot, ep_camera, ep_chassis, ep_gripper)
    approach_a(ep_robot, ep_camera, ep_chassis)
    closeclaw(ep_robot, ep_camera, ep_chassis, ep_gripper)
    carry_side(ep_robot, ep_camera, ep_chassis)
    openclaw (ep_robot, ep_camera, ep_chassis, ep_gripper)
    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel

    
