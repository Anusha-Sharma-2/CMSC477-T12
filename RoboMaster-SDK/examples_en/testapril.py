import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
import robomaster
from numpy import cos,sin,pi
from queue import Empty

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
    #roll, pitch, yaw (assume only yaw is nonzero) --------MIGHT NEED TO ADJUST THIS IF ORDER IS DIFFERENT
    seen_r = rca[1]

    robot_to_tag = get_trans_mtx(-seen_t, seen_r)

    tag_t = TAG_MAP[tag_id]["t"]
    tag_r = TAG_MAP[tag_id]["r"]
    tag_to_world = get_trans_mtx(tag_t, tag_r)

    return np.matmul(robot_to_tag,tag_to_world)


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
            # if not detection.tag_id in TAG_MAP:
            #     continue
            # tag_world_pos = np.array(TAG_MAP[detection.tag_id]) * CUBE_SIZE
            # robo_x = tag_world_pos[0] - t_ca[0]
            # robo_y = tag_world_pos[1] - t_ca[1]

            print('t_ca', t_ca)
            print('r_ca ', R_ca)
            print("Tag Id: ", detection.tag_id)
            # print("Tag position: ", tag_world_pos)
            # print("robotx: ", robo_x / CUBE_SIZE)
            # print("roboty: ", robo_y / CUBE_SIZE)
            
            
            robot_to_world = get_robo_to_world(detection.tag_id, t_ca, R_ca)
            print("Grid numbers ", robot_to_world[3,:3]/CUBE_SIZE)
            robo_x, robo_y, robo_z = robot_to_world[3,:3]/CUBE_SIZE
            
            #adjust for pytha
            robo_y -= 1
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

# if __name__ == '__main__':
#     # More legible printing from numpy.
#     robomaster.config.ROBOT_IP_STR="192.168.50.114"
#     np.set_printoptions(precision=3, suppress=True, linewidth=120)

#     ep_robot = robot.Robot()
#     ep_robot.initialize(conn_type="sta")#(conn_type="sta", sn="3JKCH7T00100J0")
#     ep_chassis = ep_robot.chassis
#     ep_camera = ep_robot.camera
#     ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

#     K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
#     marker_size_m = 0.153 # Size of the AprilTag in meters
#     apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

#     try:
#         detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag)
#     except KeyboardInterrupt:
#         pass
#     except Exception as e:
#         print(traceback.format_exc())
#     finally:
#         print('Waiting for robomaster shutdown')
#         ep_camera.stop_video_stream()
#         ep_robot.close()

def run_maze():
    ep_robot = robot.Robot()
    robomaster.config.ROBOT_IP_STR="192.168.50.114"
    ep_robot.initialize(conn_type="sta")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False)


    grid = np.zeros((12, 12))
    # TODO: Fill in obstacles (this is just top wall)
    grid[1:3, 2:10] = 1
    
    
    # start / goal pose
    start_node = (5, 1)
    goal_node = (5, 11)
    
    # call dijkstra
    #path, _ = dijkstra(grid, start_node, goal_node)
    # if not path:
    #     print("No path found")
    #     return
    path = [(3, 0), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (4, 3), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (5, 8), (5, 9), (4, 9), (3, 9), (3, 10)]
    print(path)
    
    # initialize detector
    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)    
    
    at_goal = False
    # big loop that goes until the goal
    while not at_goal:
        # for each path pair in path
        for subpath in path:
            print(subpath)
            # target_x, target_y = subpath
            # target_x *= CUBE_SIZE
            # target_y *= CUBE_SIZE
            
            at_waypoint = False
            iterations = 0
            while not at_waypoint:
                try:
                    img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                except Empty:
                    time.sleep(0.001)
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray.astype(np.uint8)
                detections = apriltag.find_tags(gray)
                robo_x, robo_y = 0,0
                   
                if len(detections) > 0:
                    #assert len(detections) == 1 # Assume there is only one AprilTag to track
                    detection = detections[0]

                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    #print('t_ca', t_ca)
                    
                    if not detection.tag_id in TAG_MAP:
                        continue
                    # tag_world_pos = np.array(TAG_MAP[detection.tag_id]) * CUBE_SIZE
                    # robo_x = t_ca[2] - tag_world_pos[0]
                    # robo_y = t_ca[0] - tag_world_pos[1]
                                
                    robot_to_world = get_robo_to_world(detection.tag_id, t_ca, R_ca)
                    print("Grid numbers ", robot_to_world[3,:3]/CUBE_SIZE)
                    
                    # robot cube coordinates
                    robo_x, robo_y, _ = robot_to_world[3,:3]/CUBE_SIZE
                    
                    #adjust for pytha
                    robo_y -= CUBE_SIZE
                    
                    
                    # velo_x = (target_x - robo_x)
                    # velo_y = (target_y - robo_y) 
                    
                    # if the di
                    # 
                    # stance between position is < 0.1, move to next path
                    #distance = np.sqrt(velo_x**2 + velo_y**2)
                    direction_x, direction_y = subpath[0] - robo_x, subpath[1] - robo_y
                    
                    # left
                    # if direction_y < 0:
                    #     print("MOVING LEFT: ", direction_x, direction_y, subpath)
                    #     ep_chassis.drive_speed(x=0, y=-0.3, z=0, timeout=5)
                    #     time.sleep(0.876)
                    # # right
                    # elif direction_y > 0:
                    #     print("MOVING RIGHT: ", direction_x, direction_y, subpath)
                    #     ep_chassis.drive_speed(x=0, y=0.3, z=0, timeout=5)
                    #     time.sleep(0.876)
                    # # forward
                    # elif direction_x > 0:
                    #     print("MOVING FORWARD: ", direction_x, direction_y, subpath)
                    #     ep_chassis.drive_speed(x=0.3, y=0, z=0, timeout=5)
                    #     time.sleep(0.876)
                    # # down
                    # elif direction_x < 0:
                    #     print("MOVING BACKWARD: ", direction_x, direction_y, subpath)
                    #     ep_chassis.drive_speed(x=0.3, y=0, z=0, timeout=5)
                    #     time.sleep(0.876)
                    # # if distance < 0.5:
                    # #     at_waypoint = True
                    # #     continue
                    # if iterations % 10 == 0:
                    #     print("NEW ITERATION")
                    #     print("Tag Id: ", detection.tag_id)
                    #     print("Grid numbers ", robot_to_world[3,:3]/CUBE_SIZE)
                    #     print("robotx: ", robo_x)
                    #     print("roboty: ", robo_y)
                    #     # print("distance: ", distance)
                    #     # print("curr goal: ", target_x/CUBE_SIZE)
                    #     # print("curr_goal: ", target_y/CUBE_SIZE)
                        
                    # # drive!
                    # ep_chassis.drive_speed(x=velo_x*0.3, y=velo_y*0.3, z=0, timeout=5)
                    # cv2.imshow("img", img)
                    # if cv2.waitKey(1) == ord('q'):
                    #     break
                
                
                
                    # stance between position is < 0.1, move to next path
                    #distance = np.sqrt(velo_x**2 + velo_y**2)
                direction_x, direction_y = subpath[0] - robo_x, subpath[1] - robo_y
                    
                # left
                if direction_y < 0:
                    print("MOVING LEFT: ", direction_x, direction_y, subpath)
                    ep_chassis.drive_speed(x=0, y=-0.3, z=0, timeout=5)
                    time.sleep(0.876)
                # right
                elif direction_y > 0:
                    print("MOVING RIGHT: ", direction_x, direction_y, subpath)
                    ep_chassis.drive_speed(x=0, y=0.3, z=0, timeout=5)
                    time.sleep(0.876)
                # forward
                elif direction_x > 0:
                    print("MOVING FORWARD: ", direction_x, direction_y, subpath)
                    ep_chassis.drive_speed(x=0.3, y=0, z=0, timeout=5)
                    time.sleep(0.876)
                # down
                elif direction_x < 0:
                    print("MOVING BACKWARD: ", direction_x, direction_y, subpath)
                    ep_chassis.drive_speed(x=0.3, y=0, z=0, timeout=5)
                    time.sleep(0.876)
                # if distance < 0.5:
                    #     at_waypoint = True
                    #     continue
                if iterations % 10 == 0:
                    print("NEW ITERATION")
                    print("Tag Id: ", detection.tag_id)
                    print("Grid numbers ", robot_to_world[3,:3]/CUBE_SIZE)
                    print("robotx: ", robo_x)
                    print("roboty: ", robo_y)
                        # print("distance: ", distance)
                        # print("curr goal: ", target_x/CUBE_SIZE)
                        # print("curr_goal: ", target_y/CUBE_SIZE)
                        
                    # drive!
                    # ep_chassis.drive_speed(x=velo_x*0.3, y=velo_y*0.3, z=0, timeout=5)
                cv2.imshow("img", img)
                if cv2.waitKey(1) == ord('q'):
                    break
                    
                    
                iterations += 1
                
                
if __name__ == "__main__":
    run_maze()
