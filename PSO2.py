#In this version, we shall update the PSO algorithm and incorporate the constriction
#factor as given by Clerc, Kennedy (2006) - A particle-swarm explosion 

import sum_gaussian
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

shape_ = (100, 100)
terrain_map_ = np.zeros(shape_)
n_particles_ = 100
current_state_ = np.zeros(shape_)
current_locations_ = np.zeros((n_particles_, 2))
current_velocities_ = np.zeros((n_particles_, 2))
maximum_velocity_ = shape_[0]/10
best_global_ = np.zeros(2)
global_optimal_value_ = 0
personal_best_ = np.zeros((n_particles_, 2))
personal_optimal_values_ = np.zeros(n_particles_)
num_iter_ = 100
vibration_ = 0.9
inertia_ = 1 #inertia_ is not to be changed from default value of 1
social_ = 2
cognition_ = 2.5
phi_ = social_ + cognition_
#chi_ is the constriction factor
chi_ = 2/np.abs(2 - phi_ - np.sqrt(phi_*phi_ - 4*phi_))
count_ = 0

def compute_objective_function(i,j):
    global terrain_map_ 
    return terrain_map_[i][j]

def initialize_particles():
    global shape_, terrain_map_, current_state_, personal_best_
    global current_locations_, current_velocities_
    global best_global_, global_optimal_value_, personal_optimal_values_
    global n_particles_, vibration_
    for i in range(0, n_particles_):
        current_locations_[i][0] = int(np.random.randint(0, shape_[0] - 1))
        current_locations_[i][1] = int(np.random.randint(0, shape_[1] - 1))
        #print(i, "The current location is: ", current_locations_[i] )
        #print(current_locations_[i])
        current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 1
        #print(current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])])
        personal_best_[i] = current_locations_[i]
        personal_optimal_values_[i] = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
        #print(i, "Personal best location is:", personal_best_[i], "Personal optimal value is:", personal_optimal_values_[i])
        if compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) > global_optimal_value_:
            global_optimal_value_ = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
            best_global_ = current_locations_[i]
        #print("Global best location is:", best_global_, "Global optimal value is:", global_optimal_value_)
        current_velocities_[i][0] = int(np.random.randint(int(-shape_[0]*vibration_), int(shape_[0]*vibration_)))
        current_velocities_[i][1] = int(np.random.randint(int(-shape_[1]*vibration_), int(shape_[1]*vibration_)))
        #print(i, "Current velocity is:", current_velocities_[i])


def velocity_update():
    global n_particles_, count_, chi_
    global current_velocities_, social_, cognition_, inertia_
    global personal_best_, best_global_, global_optimal_value_, personal_optimal_values_
    for i in range(0, n_particles_):
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)
        #print(i, r1, r2)
        #print("Iteration number:", count_)
        #print(i, "The old velocity was:", current_velocities_[i])
        #print(i, "Social distance is:", best_global_ - current_locations_[i])
        current_velocities_[i][0] = int(chi_*(inertia_ * current_velocities_[i][0] + r1*cognition_*(personal_best_[i][0] - current_locations_[i][0]) + r2*social_*(best_global_[0] - current_locations_[i][0])))
        current_velocities_[i][1] = int(chi_*(inertia_ * current_velocities_[i][1] + r1*cognition_*(personal_best_[i][1] - current_locations_[i][1]) + r2*social_*(best_global_[1] - current_locations_[i][1])))
        if current_velocities_[i][0] > maximum_velocity_: 
            current_velocities_[i][0] = maximum_velocity_
        elif current_velocities_[i][0] < -maximum_velocity_:
            current_velocities_[i][0] = -maximum_velocity_
        if current_velocities_[i][1] > maximum_velocity_: 
            current_velocities_[i][1] = maximum_velocity_
        elif current_velocities_[i][1] < -maximum_velocity_:
            current_velocities_[i][1] = -maximum_velocity_
        
        #print(i, "The new velocit is:", current_velocities_[i])

def location_update():
    global current_locations_, current_state_, current_velocities_
    global n_particles_, shape_
    global best_global_, personal_best_
    global global_optimal_value_, personal_optimal_values_

    for i in range(0, n_particles_):
        current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 0 
        #print(i, "The old location is: ", current_locations_[i])
        if current_locations_[i][0] + current_velocities_[i][0] >= shape_[0]:
            current_locations_[i][0] = shape_[0] - 1 
        elif current_locations_[i][0] + current_velocities_[i][0] <= 0:
            current_locations_[i][0] = 0
        else: current_locations_[i][0] = int(current_locations_[i][0] + current_velocities_[i][0])
        if current_locations_[i][1] + current_velocities_[i][1] >= shape_[1]:
            current_locations_[i][1] = shape_[1] - 1 
        elif current_locations_[i][1] + current_velocities_[i][1] <= 0:
            current_locations_[i][1] = 0
        else: current_locations_[i][1] = int(current_locations_[i][1] + current_velocities_[i][1])
        #print(i, "The new location is:", current_locations_[i])
        current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 1 
        if compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) > global_optimal_value_: 
            global_optimal_value_ = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
            best_global_ = current_locations_[i]
        if compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) > personal_optimal_values_[i]:
            personal_optimal_values_[i] = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
            personal_best_[i] = current_locations_[i]

def compute_fitness_function():
    global current_state_, shape_, count_
    sum = 0
    for i in range(0, shape_[0]):
        for j in range(0, shape_[1]): 
            sum = sum + compute_objective_function(i,j)*current_state_[i][j]
    print("At iteration number:", count_, "The value of the fitness function:", sum)
    print("The optimal solution was found at:", best_global_)

def summation_function():
    sum = 0
    for i in range(0, shape_[0]):
        for j in range(0, shape_[1]): 
            sum = sum + compute_objective_function(i,j)
    print("The summation across the domain is:", sum)
    
if __name__ == "__main__":
    shape_ = (100, 100)
    n = 5
    k = 2
    mean = np.zeros((n,2))
    covariance = np.zeros(n)
    for i in range(0,n):
        #print(i)
        mean[i][0] = np.random.randint(0, shape_[0])
        mean[i][1] = np.random.randint(0, shape_[1])
        covariance[i] = 1
        print(mean[i])
    sum_of_gaussians = sum_gaussian.Sum_of_Gaussians(shape_,n,mean,covariance)
    terrain_map_ = sum_of_gaussians.gaussian_sum
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x = np.linspace(0, shape_[0] - 1, shape_[0])
    y = np.linspace(0, shape_[1] - 1, shape_[1])
    #print(x,y)
    X, Y = np.meshgrid(x,y)
    #Z = compute_objective_function(X,Y)
    #ax.plot_wireframe(X,Y,Z, color = 'r', linewidth = 0.2)




    #print(np.max(terrain_map_))
    initialize_particles()
    #print(best_global_)
    for j in range(0, num_iter_):
        velocity_update()
        location_update()
        compute_fitness_function()
        count_ = count_ + 1
    summation_function()

    # ax = fig.add_subplot(111, projection = '3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # x = np.linspace(0, shape_[0] - 1, shape_[0])
    # y = np.linspace(0, shape_[1] - 1, shape_[1])
    # #print(x,y)
    # X, Y = np.meshgrid(x,y)
    # Z = terrain_map_
    # ax.plot_wireframe(X,Y,Z, color = 'r', linewidth = 0.2)
    # images = []
    # image = ax.scatter3D([current_locations_[i][0] for i in range(n_particles_)],[current_locations_[i][1] for i in range(n_particles_)],[compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) for i in range(n_particles_)], c='b')
    # images.append([image])
    # animated_image = animation.ArtistAnimation(fig, images)
    # animated_image.save('./pso2_output.gif', writer='pillow')
    # im = Image.open('./pso2_output.gif')

    x = np.linspace(0, shape_[0] - 1, shape_[0])
    y = np.linspace(0, shape_[1] - 1, shape_[1])
    X, Y = np.meshgrid(x,y)
    Z = terrain_map_
    fig, ax = plt.subplots(figsize=(8,6))
    img = ax.imshow(Z, extent=[0, shape_[0] - 1, 0, shape_[1] - 1], origin='lower', cmap='viridis', alpha=0.5)
    p_plot = ax.scatter(current_locations_[:,0], current_locations_[:,1], marker='o', color='blue', alpha=0.5)
    plt.show()