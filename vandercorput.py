##### Implementation using Vander Corput Sequence

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon, Point
import numpy as np
import math

def van_der_corput(n, base=2):
    result = 0
    f = 1.0 / base
    i = 1
    while n > 0:
        result += (n % base) * f
        n //= base
        f /= base
        i *= base
    return result

def generate_evenly_distributed_points(x, N):
    # Calculate the interval between points using van der Corput sequence
    points = []
    for i in range(N):
        x_coord = van_der_corput(i, base=2) * x
        y_coord = van_der_corput(i, base=3) * x
        points.append((x_coord, y_coord))
    return points

def generate_evenly_distributed_points_with_perturbation(x, N, theta):
    # Calculate the interval between points using van der Corput sequence
    lambda_0 = 10
    points = []
    for i in range(N):
        x_coord = van_der_corput(i, base=2) * x + lambda_0*math.cos(theta)
        y_coord = van_der_corput(i, base=3) * x + lambda_0*math.sin(theta)
        points.append((x_coord, y_coord))
    return points


num_plots = 10
for i in range(1,num_plots):
  # Size of the square
  x = 100  # Replace with your desired size
  # Number of points to generate
  N = 100  # Replace with your desired number of points

  theta = 2*math.pi*np.random.random()
  # Generate evenly distributed points
  #points_new = generate_evenly_distributed_points(x, N)
  points_new = generate_evenly_distributed_points_with_perturbation(x, N, theta)

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
  #seed_points = points_new 
  combined_points = np.vstack((polygon_coords, seed_points))

  vor2 = Voronoi(seed_points)
  print("plotted ", len(seed_points), " points")

  # Plot the Voronoi diagram
  voronoi_plot_2d(vor2, show_vertices=False, show_points=True)

  # Plot the original polygon
  polygon_coords = np.append(polygon_coords, [polygon_coords[0]], axis=0)
  plt.plot(polygon_coords[:, 0], polygon_coords[:, 1], 'k-')

  # Set axis limits and labels
  plt.xlim(0, 100)  # Adjust this based on your polygon dimensions
  plt.ylim(0, 100)  # Adjust this based on your polygon dimensions
  plt.gca().set_aspect('equal', adjustable='box')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.show()
