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

print('model')
model = YOLO("best.pt")

# Use vid instead of ep_camera to use your laptop's webcam
vid = cv2.VideoCapture(0)
    
robomaster.config.ROBOT_IP_STR="192.168.50.114"
ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta")
np.set_printoptions(precision=3, suppress=True, linewidth=120)
ep_chassis = ep_robot.chassis
ep_camera = ep_robot.camera
ep_gripper = ep_robot.gripper
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)



frame_count = 0

# def center_robot(ep_robot, ep_chassis, ep_camera, apriltag, box_center):
    
#     while box_center - IMAGE_CENTER_X <= 2:
#         try:
#             img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
#         except Empty:
#             continue
        
#         detections = apriltag.find_tags(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#         ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5) # Scan for tag
        
#         if len(detections) > 0:
#             det = detections[0]
#             if det.tag_id == 32:
#                 _, R_ca = get_pose_apriltag_in_camera_frame(det)
#                 x_axis = R_ca[:,0]
#                 robot_yaw = np.rad2deg(np.arctan2(x_axis[2], x_axis[0]))
                
#                 if abs(robot_yaw) < 1:
#                     ep_chassis.drive_speed(x=0, y=0, z=0)
#                     print("Centered around tag, starting execution.")
#                     break
#                 ep_chassis.drive_speed(x=0, y=0, z=-robot_yaw, timeout=5)
#                 time.sleep(1)

def move_forward(ep_robot, ep_chassis, ep_camera, distance):
    while distance >= 0.27:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue
        ep_chassis.drive_speed(x=0.1, y=0, z=0, timeout=5)
    return

alpha = 0.85
def low_pass_filter(prev_val, new_val):
    return alpha * new_val + (1 - alpha) * prev_val
# drop claw code
# Open the gripper
ep_gripper.open(power=50)
time.sleep(1)
ep_gripper.pause()

# Move the arm to the "retracted" position
ep_robot.robotic_arm.moveto(x=100, y=30).wait_for_completed()
# Move the arm forward and down in order to pickup an object
# (we do this in two moves to avoid the "keep out zone" where the robot may hit itself)
ep_robot.robotic_arm.moveto(x=180, y=30).wait_for_completed()
ep_robot.robotic_arm.moveto(x=180, y=-50).wait_for_completed()

while frame_count < 300:
    #ret, frame = vid.read()
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if frame is not None:
            start = time.time()
            if model.predictor:
                model.predictor.args.verbose = False
            result = model.predict(source=frame, show=False)[0]

            #DIY visualization is much faster than show=True for some reason
            boxes = result.boxes
            distance_z = 0
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cv2.rectangle(frame,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
            
                # get box coordinates
                xyxy = box.xyxy.cpu().numpy().flatten()
                x1, y1, x2, y2 = xyxy
                box_center_x = (x1 + x2) / 2
                box_height_pixels = y2 - y1

                REAL_LEGO_HEIGHT_M = 0.138
                FOCAL_LENGTH = 314.0 

                # Z = object height / (pixels / camera focal length)
                curr_distance_z = REAL_LEGO_HEIGHT_M / (box_height_pixels / FOCAL_LENGTH)
                distance_z = low_pass_filter(distance_z, curr_distance_z)

                # calculate error
                heading_error = IMAGE_CENTER_X - box_center_x

                print(f"Block is {distance_z:.2f}m away. Heading error: {heading_error}px")
                
                # move that set amount
                if distance_z < 0.22:
                    # Close the gripper on the object
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    ep_gripper.close(power=50)
                    print(distance_z)
                    time.sleep(1)
                    ep_gripper.pause()
                    break
                    
                ep_chassis.drive_speed(x=0.1, y=0, z=0)
                
                
            cv2.imshow('frame', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except Empty:
        time.sleep(0.001)
        continue
        
    
ep_camera.stop_video_stream()
ep_robot.close()