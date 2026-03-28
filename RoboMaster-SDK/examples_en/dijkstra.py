import numpy as np
import heapq
#import matplotlib.pyplot as plt
#from matplotlib.animation import PillowWriter

def dijkstra(grid, start, goal):
    
    rows, cols = grid.shape
    # initialize priority queue with cost + current_node
    pq = [(0, start)]
    dist = {start: 0}
    
    # parent keeps track of the path for 
    parent = {start: None}
    visit_order = []

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while pq:
        d, current = heapq.heappop(pq)

        if current in visit_order: continue
        visit_order.append(current)
        
        if current == goal:
            return reconstruct_path(parent, current), visit_order

        if d > dist.get(current, float('inf')):
            continue

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check boundaries and obstacles
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # wall
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                
                new_dist = d + 1
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return None, visit_order

# reconstruct path
def reconstruct_path(parent, current):
    path = []
    while current:
        path.append(current)
        current = parent[current]
    return path[::-1] 

if __name__ == '__main__':
    grid = np.genfromtxt('Map1.csv', delimiter=',')
    start_pos = np.argwhere(grid == 2)
    goal_pos = np.argwhere(grid == 3)
    
    if start_pos.size > 0 and goal_pos.size > 0:
        start = tuple(start_pos[0])
        goal = tuple(goal_pos[0])

        path, visited = dijkstra(grid, start, goal)

        # Animation Setup
        writer = PillowWriter(fps=30)
        fig = plt.figure(figsize=(8, 8))
        
        obstacles = np.argwhere(grid == 1)
        obs_x, obs_y = obstacles[:, 1], obstacles[:, 0]
    
        if path:
            print(f"Path length: {len(path)}")
            print("Path coordinates:", path)

            # animation video
            with writer.saving(fig, "dijkstra_navigation.gif", 100):
                    for i in range(0, len(visited), 5):
                        plt.clf()
                        # Plot obstacles
                        plt.scatter(obs_x, obs_y, color='black', marker='s', s=1)
                        # Plot visited nodes
                        vis_nodes = np.array(visited[:i])
                        if len(vis_nodes) > 0:
                            plt.scatter(vis_nodes[:, 1], vis_nodes[:, 0], color='blue', s=2, alpha=0.3)
                        
                        plt.title("Dijkstra")
                        plt.xlim(0, 100); plt.ylim(0, 100)
                        writer.grab_frame()

                    # draw final path
                    path_coords = np.array(path)
                    plt.plot(path_coords[:, 1], path_coords[:, 0], color='red', linewidth=2)
                    # stop for path
                    for _ in range(60):
                        writer.grab_frame()
        else:
            print("No path found")
    else:
        print("No goal or path")
    