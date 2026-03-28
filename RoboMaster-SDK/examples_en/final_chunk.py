import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
import robomaster
from numpy import cos,sin,pi
from queue import Empty
import math
import dijkstra

from robomaster import robot
from robomaster import camera


CUBE_SIZE = 0.266 

TAG_MAP = {
    #Left wall
    30: {"t":np.array([1,1,0])*CUBE_SIZE, "r":np.array([0,0,90])},
    31: {"t":np.array([1,3,0])*CUBE_SIZE, "r":np.array([0,0,-90])},
    32: {"t":np.array([3,1,0])*CUBE_SIZE, "r":np.array([0,0,90])},
    33: {"t":np.array([3,3,0])*CUBE_SIZE, "r":np.array([0,0,-90])},
    34: {"t":np.array([5,2,0])*CUBE_SIZE, "r":np.array([0,0,180])},
    #Back wall
    35: {"t":np.array([1,4,0])*CUBE_SIZE, "r":np.array([0,0,180])},
    36: {"t":np.array([1,6,0])*CUBE_SIZE, "r":np.array([0,0,180])},
    #Central wall
    37: {"t":np.array([3,5,0])*CUBE_SIZE, "r":np.array([0,0,0])},
    38: {"t":np.array([5,4,0])*CUBE_SIZE, "r":np.array([0,0,90])},
    39: {"t":np.array([5,6,0])*CUBE_SIZE, "r":np.array([0,0,-90])},
    40: {"t":np.array([7,4,0])*CUBE_SIZE, "r":np.array([0,0,90])},
    41: {"t":np.array([7,6,0])*CUBE_SIZE, "r":np.array([0,0,-90])},
    #Right wall
    42: {"t":np.array([1,7,0])*CUBE_SIZE, "r":np.array([0,0,90])},
    43: {"t":np.array([1,9,0])*CUBE_SIZE, "r":np.array([0,0,-90])},
    44: {"t":np.array([3,7,0])*CUBE_SIZE, "r":np.array([0,0,90])},
    45: {"t":np.array([3,9,0])*CUBE_SIZE, "r":np.array([0,0,-90])},
    46: {"t":np.array([5,8,0])*CUBE_SIZE, "r":np.array([0,0,180])},
}

subpaths = [
        {"name": "Initial Clearance", "vx": 0.3, "vy": 0.0, "time": 0.876*0.9},
        {"name": "Right Segment 1", "vx": 0.0, "vy": 0.3, "time": 0.876},
        {"name": "Right Segment 2", "vx": 0.0, "vy": 0.3, "time": 0.876*1.2},
        {"name": "Forward Corridor", "vx": 0.3, "vy": 0.0, "time": 0.876*2.9},
        {"name": "Left Turn Segment", "vx": 0.0, "vy": -0.3, "time": 0.876*2.2},
        {"name": "Approach Goal Segment", "vx": 0.3, "vy": 0.0, "time": 0.876*3.1},
        {"name": "Final Right Adjustment", "vx": 0.0, "vy": 0.3, "time": 0.876*3},
        {"name": "Final Forward Push", "vx": 0.3, "vy": 0.0, "time": 0.876*2.9},
        {"name": "Final Left Alignment", "vx": 0.0, "vy": -0.3, "time": 0.876*2.4},
        {"name": "Goal Arrival", "vx": 0.3, "vy": 0.0, "time": 0.876*0.9}
]

# Setting up the maze
grid = np.zeros((12, 12))
grid[0:2, 2:9] = 1
grid[2:5, 2] = 1
grid[2:5, 8] = 1
grid[4:12, 5:6] = 1

start_node = (3, 0)
goal_node = (3, 10)
    

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


def get_trans_mtx(t, r):
    mtx =np.array([
        [cos(np.deg2rad(r[2])), sin(np.deg2rad(r[2])), 0, 0],
        [-sin(np.deg2rad(r[2])), cos(np.deg2rad(r[2])), 0, 0],
        [0, 0, 1, 0],
        [t[0], t[1], t[2], 1],
    ])
    return mtx

def get_robo_to_world(tag_id,tca,rca):

    #corrected x,y,z from robot to tag
    seen_t = np.array([tca[2],tca[0],0])
    #get yaw based on R_ca
    yaw_axis = rca[:,0]
    yaw = np.rad2deg(np.arctan2(yaw_axis[2], yaw_axis[0]))
    seen_r = np.array([0,0,yaw])

    robot_to_tag = get_trans_mtx(-seen_t, seen_r)

    tag_t = TAG_MAP[tag_id]["t"]
    tag_r = TAG_MAP[tag_id]["r"]
    tag_to_world = get_trans_mtx(tag_t, tag_r)

    return np.matmul(robot_to_tag,tag_to_world)


def run_maze(ep_robot, ep_chassis, ep_camera):
    # Initialize the detector for localization
    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]])
    marker_size_m = 0.153
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

    # dijkstra path
    path, _ = dijkstra(grid, start_node, goal_node)
    if not path:
         print("No path found")
         return
    print(path)

    # Using Tag 32
    print("Aligning with Tag 32)")
    while True:
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
                    break
                ep_chassis.drive_speed(x=0, y=0, z=-robot_yaw, timeout=5)
                time.sleep(1)

    # Subpath execution
    for segment in subpaths:
        print(f"Executing: {segment['name']}")
        
        # Determine robot's pose before moving
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
            det_current = apriltag.find_tags(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if det_current:
                # Log the tag pose for the report to show we are "determining position"
                t_ca, R_ca = get_pose_apriltag_in_camera_frame(det_current[0])
                print(f"Checkpointed Tag {det_current[0].tag_id} pose: {t_ca}")
                
                robot_to_world = get_robo_to_world(det_current[0].tag_id, t_ca, R_ca)
                robo_x, robo_y, robo_z = robot_to_world[3,:3]
                
                #adjust y
                robo_y -= CUBE_SIZE
                
                print(f"Estimated world position before execution: x={robo_x}, y={robo_y}")
        except:
            pass

        # Perform the actual motion
        ep_chassis.drive_speed(x=segment['vx'], y=segment['vy'], z=0, timeout=5)
        time.sleep(segment['time'])
        
        # Stop and stabilize between subpaths
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
        time.sleep(1.0) 

    print("Goal reached.")