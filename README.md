# Robomaster EP: Perception, Planning, and Navigation

This repository contains a full robotics navigation stack developed for the **Robomaster EP** platform. The project integrates computer vision (AprilTags), graph-based path planning (Dijkstra, BFS, DFS), and autonomous trajectory execution in a structured environment.

## Overview
The system is designed to navigate a robot through a maze of 26.6 x 26.6 cm storage cubesusing markers for localization.

### Main files
* `apriltag.py`: Real-time vision pipeline that detects `tag36h11` AprilTags to estimate the robot's 6D pose relative to landmarks
* `final_chunk.py`: A sequence-based maze solver that executes decomposed path segments while maintaining yaw alignment via visual feedback.

## Videos
* [AprilTag Tracking](https://drive.google.com/file/d/1ZUQxZYmTbaDLPrsYjdEvqS5sQyU4rIg0/view?usp=sharing)
* [Maze Solving](https://drive.google.com/file/d/1bICBGcefuNO0uB_EbMopj4_kwtz1Diqc/view?usp=drive_link)
