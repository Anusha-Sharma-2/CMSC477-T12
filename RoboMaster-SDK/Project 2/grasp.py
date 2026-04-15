import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
import robomaster
from numpy import cos,sin,pi
from queue import Empty
import math

from robomaster import robot
from robomaster import camera


REAL_LEGO_HEIGHT_M = 0.2
FOCAL_LENGTH = 314
IMAGE_CENTER_X = 320

class AprilTagDetector:
    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)

    def find_tags(self, frame_gray):
        detections = self.detector.detect(frame_gray, estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.marker_size_m)
        return detections

def get_pose_apriltag_in_camera_frame(detection):
    R_ca = detection.pose_R
    t_ca = detection.pose_t
    return t_ca.flatten(), R_ca

def draw_detections(frame, detections):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)
        ep_chassis.drive_speed(x=-0, y=-0, z=0, timeout=5)
        
        
        if len(detections) > 0:
            #assert len(detections) == 1 # Assume there is only one AprilTag to track
            detection = detections[0]

            t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)

            print('t_ca', t_ca)
            print('r_ca ', R_ca)
            print("Tag Id: ", detection.tag_id)

            # We started our work here for the demo
            # for bonus - we define the distance to be 1m from the april tag
            goal_pose = [0, 0, 1]
            
            # unpacking the t_ca
            z = float(t_ca[2])
            x = float(t_ca[0])
            
            
            # x is forward/backward
            # y is left/right
            # tca - z is forward/backward
            # tca - x is left/right
            
            tracking_error = t_ca - goal_pose
            print("tracking error:", tracking_error)
            xe,_, ze = tracking_error
            
            #ep_chassis.drive_speed(x=0.3*ze, y=xe, z=0, timeout=5)

        draw_detections(img, detections)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
        

# NOW OUR CODE EXISTS
def sub_data_handler(sub_info):
    pos_x, pos_y = sub_info

    # Fix the integer overflow in pos_y
    if pos_y > 2**31 - 1:
        pos_y = pos_y - 2**32

    # You can use these values to confirm the robot arm is where it is supposed to be
    # It is also usable for determine the right setpoints to send to "moveto" commands
    print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))

# center the object in front of the robot
def center_robot(ep_robot, ep_chassis, ep_camera, box_center):
    
    while box_center - IMAGE_CENTER_X >= 2:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        
        detections = apriltag.find_tags(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5) # Scan for tag
        
        if len(detections) > 0:
            det = detections[0]
            if det.tag_id == 32:
                _, R_ca = get_pose_apriltag_in_camera_frame(det)
                x_axis = R_ca[:,0]
                robot_yaw = np.rad2deg(np.arctan2(x_axis[2], x_axis[0]))
                
                if abs(robot_yaw) < 1:
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    print("Centered around tag, starting execution.")
                    return
                ep_chassis.drive_speed(x=0, y=0, z=-robot_yaw, timeout=5)
                time.sleep(1)
        

if __name__ == '__main__':
    robomaster.config.ROBOT_IP_STR="192.168.50.114"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    
    # create apriltag project
    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

    # Start printing the gripper position
    ep_arm.sub_position(freq=5, callback=sub_data_handler)

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

    
    # Estimate aruco distance:
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)
        ep_chassis.drive_speed(x=-0, y=-0, z=0, timeout=5)
        
        
        if len(detections) > 0:
            #assert len(detections) == 1 # Assume there is only one AprilTag to track
            detection = detections[0]
            
            # get the four corners
            pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

            img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            top_left = pts[0][0]  # First corner
            top_left_x, top_left_y = top_left
            top_right = pts[1][0] # Second corner
            top_right_x, _ = top_right
            bottom_right = pts[2][0]  # Third corner
            bottom_left = pts[3][0] # Fourth corner
            _, bottom_left_y = bottom_left
            
            # height and center
            print("top left", top_left)
            height = top_left_y - bottom_left_y
            print("height", height)
            center = int((top_left_x + top_right_x)/2)
            print("center", center)
        
            # Z = object height / (pixels / camera focal length)
            distance_z = REAL_LEGO_HEIGHT_M / (height / FOCAL_LENGTH)

            # calculate error
            heading_error = IMAGE_CENTER_X - center

            print(f"Block is {distance_z:.2f}m away. Heading error: {heading_error}px")

            t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)

            print('t_ca', t_ca)
            print('r_ca ', R_ca)
            print("Tag Id: ", detection.tag_id)
            
            # We started our work here for the demo
            # for bonus - we define the distance to be 1m from the april tag
            goal_pose = [0, 0, 1]
            
            # unpacking the t_ca
            z = float(t_ca[2])
            x = float(t_ca[0])
            
            
            # x is forward/backward
            # y is left/right
            # tca - z is forward/backward
            # tca - x is left/right
            
            tracking_error = t_ca - goal_pose
            print("tracking error:", tracking_error)
            xe,_, ze = tracking_error
            
            # center robot to tag
            center_robot(ep_robot, ep_chassis, ep_camera, center)
            
            #ep_chassis.drive_speed(x=0.3*ze, y=xe, z=0, timeout=5)
    # Close the gripper on the object
    ep_gripper.close(power=50)
    time.sleep(1)
    ep_gripper.pause()

    # Lift the object in the gripper
    ep_robot.robotic_arm.moveto(x=180, y=100).wait_for_completed()

    # Relative move commands work inconsistently but if you want to use them they are:
    # ep_arm.move(x=50).wait_for_completed()
    # ep_arm.move(x=-50).wait_for_completed()
    # ep_arm.move(y=50).wait_for_completed()
    # ep_arm.move(y=-50).wait_for_completed()

    ep_robot.close()