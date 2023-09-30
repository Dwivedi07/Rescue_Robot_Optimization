import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random 
from shapely.geometry import Polygon, Point
np.random.seed(69)
import numpy as np

def generate_evenly_distributed_points(x, N):
    # Calculate the interval between points
    interval = x / np.sqrt(N)

    # Initialize a list to store the points as tuples
    points = []

    # Generate evenly distributed points
    for i in range(int(np.sqrt(N))):
        for j in range(int(np.sqrt(N))):
            x_coord = i * interval + interval / 2
            y_coord = j * interval + interval / 2
            points.append((x_coord, y_coord))

    return points

# Size of the square
x = 100  # Replace with your desired size

# Number of points to generate
N = 200  # Replace with your desired number of points

# Generate evenly distributed points
points_new = generate_evenly_distributed_points(x, N)

original_polygon_coords = [(5, 5), (95, 5), (95, 25), (25, 25), (95, 75), (95, 95), (5, 95), (5, 75), (75, 75), (5, 25)]
polygon = Polygon(original_polygon_coords)

# Shuffle the points randomly
random.shuffle(points_new)

# Initialize a list to store points inside the polygon
selected_points = []

# Iterate through the shuffled points and select p points inside the polygon
p = 20 # Replace with your desired number of points
for point in points_new:
    if polygon.contains(Point(point)):
        selected_points.append(point)
        if len(selected_points) == p:
            break

# Define the original polygon coordinates
polygon_coords = [(5, 5), (95, 5), (95, 25), (25, 25), (95, 75), (95, 95), (5, 95), (5, 75), (75, 75), (5, 25)]

# Define seed points within a specific region to create smaller subspaces
seed_points = selected_points
# Combine the original polygon coordinates and the seed points
combined_points = np.vstack((polygon_coords, seed_points))

# Compute the Voronoi diagram
vor = Voronoi(combined_points)

# Plot the Voronoi diagram
voronoi_plot_2d(vor, show_vertices=False, show_points=True)

# Plot the original polygon
polygon_coords = np.append(polygon_coords, [polygon_coords[0]], axis=0)
plt.plot(polygon_coords[:, 0], polygon_coords[:, 1], 'k-')

# Set axis limits and labels
plt.xlim(0, 100)  # Adjust this based on your polygon dimensions
plt.ylim(0, 100)  # Adjust this based on your polygon dimensions
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Voronoi Diagram with Smaller Subspaces')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()
