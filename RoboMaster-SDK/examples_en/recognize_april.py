import numpy as np
import time
import cv2
from dijkstra import dijkstra
from apriltag import AprilTagDetector, get_pose_apriltag_in_camera_frame, detect_tag_loop
from robomaster import robot


CUBE_SIZE = 0.266 

TAG_MAP = {
    30: (2, 2), 31: (2, 3), 32: (4, 2), 33: (4, 3),
    34: (5, 2), 35: (1, 5), 36: (1, 7), 37: (4, 6),
    38: (5, 5), 39: (5, 7), 40: (7, 5), 41: (7, 7),
    42: (2, 8), 43: (2, 9), 44: (4, 8), 45: (4, 9)
}
     
def run_maze():
    ep_robot = robot.Robot()
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
    path, _ = dijkstra(grid, start_node, goal_node)
    if not path:
        print("No path found")
        return
    path = [(4, 1), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4), (5, 4), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (6, 8), (6, 9), (6, 10), (5, 10), (4, 10), (4, 11)]
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
            target_x, target_y = subpath
            target_x *= CUBE_SIZE
            target_y *= CUBE_SIZE
            
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
                     
                if len(detections) > 0:
                    #assert len(detections) == 1 # Assume there is only one AprilTag to track
                    detection = detections[0]

                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    #print('t_ca', t_ca)
                    
                    if not detection.tag_id in TAG_MAP:
                        continue
                    tag_world_pos = np.array(TAG_MAP[detection.tag_id]) * CUBE_SIZE
                    robo_x = t_ca[2] - tag_world_pos[0]
                    robo_y = t_ca[0] - tag_world_pos[1]
                    
                    if iterations % 10 == 0:
                        print("NEW ITERATION")
                        print("Tag Id: ", detection.tag_id)
                        print("Tag position: ", tag_world_pos)
                        print("robotx: ", robo_x / CUBE_SIZE)
                        print("roboty: ", robo_y / CUBE_SIZE)
                    
                    velo_x = (target_x - robo_x)
                    velo_y = (target_y - robo_y) 
                    
                    # if the distance between position is < 0.1, move to next path
                    distance = np.sqrt(velo_x**2 + velo_y**2)
                    if distance < 0.1:
                        at_waypoint = True
                        continue
                    # drive!
                    ep_chassis.drive_speed(x=velo_x*0.1, y=velo_y*0.1, z=0, timeout=5)
                    cv2.imshow("img", img)
                    if cv2.waitKey(1) == ord('q'):
                        break
                    
                iterations += 1
                
                
if __name__ == "__main__":
    run_maze()