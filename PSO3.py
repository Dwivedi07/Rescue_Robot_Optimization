#In this version we shall be implementing the MSPSO algorithm, which will make it
#possible to arrive at local maxima in addition to global maxima
#this will require us to define an additional speciation function. 
#Also, we are defining the number of seeds manually here.  



import sum_gaussian
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

shape_ = (100, 100)
terrain_map_ = np.zeros(shape_)
n_particles_ = 100
markers_ = np.zeros(n_particles_) 
found_ = True
n_seeds_ = 0
particle_belongs_to_species_ = np.zeros(n_particles_)
n_detected_ = 0 
sigma_ = shape_[0]/2.5
current_state_ = np.zeros(shape_)
current_locations_ = np.zeros((n_particles_, 2))
current_velocities_ = np.zeros((n_particles_, 2))
maximum_velocity_ = shape_[0]/10
best_global_ = np.zeros(2)
best_particle_ = 0
global_optimal_value_ = 0
personal_best_ = np.zeros((n_particles_, 2))
personal_optimal_values_ = np.zeros(n_particles_)
species_seeds_ = []
seed_locations_ = []
num_iter_ = 100
inertia_ = 0.01
social_ = 2
cognition_ = 1
count_ = 0
sum_fitness_ = 0
vibration_ = 0.8 

def compute_objective_function(i,j):
    global terrain_map_ 
    int_i = int(i)
    int_j = int(j)
    return terrain_map_[int_i][int_j]

def initialize_particles():
    global shape_, terrain_map_, current_state_, personal_best_
    global current_locations_, current_velocities_
    global best_global_, global_optimal_value_, personal_optimal_values_
    global n_particles_, vibration_, best_particle_
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
            best_particle_ = i
        #print("Global best location is:", best_global_, "Global optimal value is:", global_optimal_value_)
        current_velocities_[i][0] = int(np.random.randint(int(-shape_[0]*vibration_), int(shape_[0]*vibration_)))
        current_velocities_[i][1] = int(np.random.randint(int(-shape_[1]*vibration_), int(shape_[1]*vibration_)))
        #print(i, "Current velocity is:", current_velocities_[i])

def speciation():
    global shape_, terrain_map_, current_state_, personal_best_
    global current_locations_, current_velocities_
    global best_global_, global_optimal_value_, personal_optimal_values_
    global n_particles_, vibration_, n_detected_, sigma_
    global n_seeds_, found_, species_seeds_, markers_, best_particle_
    global particle_belongs_to_species_, seed_locations_
    while(np.sum(markers_) < 100):
        temp_best_value = 0
        temp_best_is = 0
        for i in range(0, n_particles_):
            if (markers_[i] == 0 and compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) >= temp_best_value ):
                temp_best_is = i
                #print("The temporary best particle is:", temp_best_is)
                temp_best_value = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
        markers_[temp_best_is] = 1 
        found_ = False
        for i in range(0, n_seeds_):
            if np.sqrt((seed_locations_[i][0] - current_locations_[int(temp_best_is)][0])**2 + (seed_locations_[i][1] - current_locations_[int(temp_best_is)][1])**2) < sigma_:
                found_ = True
                #print("The temporary best is:", temp_best_is)
                particle_belongs_to_species_[int(temp_best_is)] = i
                break
        if (not found_):
            species_seeds_.append(temp_best_is)
            seed_locations_.append([int(current_locations_[int(temp_best_is)][0]), int(current_locations_[int(temp_best_is)][1])])
            n_seeds_ = n_seeds_ + 1
        print("The sum of the marker matrix:", np.sum(markers_))
        #print("Markers:", markers_)
        #print(species_seeds_)
        #print(seed_locations_)
    print("The number of seeds is:", n_seeds_)
    print("The particle species array is", particle_belongs_to_species_)


def velocity_update():
    global n_particles_, count_
    global current_velocities_, social_, cognition_, inertia_
    global personal_best_, best_global_, global_optimal_value_, personal_optimal_values_
    global n_particles_, vibration_, n_detected_, sigma_
    global n_seeds_, found_, species_seeds_, markers_, best_particle_
    global particle_belongs_to_species_, seed_locations_
    for i in range(0, n_particles_):
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)
        #print(i, r1, r2)
        #print("Iteration number:", count_)
        #print(i, "The old velocity was:", current_velocities_[i])
        #print(i, "Social distance is:", best_global_ - current_locations_[i])
        current_velocities_[i][0] = int(inertia_ * current_velocities_[i][0] + r1*cognition_*(personal_best_[i][0] - current_locations_[i][0]) + r2*social_*(best_global_[0] - current_locations_[i][0]))
        current_velocities_[i][1] = int(inertia_ * current_velocities_[i][1] + r1*cognition_*(personal_best_[i][1] - current_locations_[i][1]) + r2*social_*(best_global_[1] - current_locations_[i][1]))
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
    global current_state_, shape_, count_, sum_fitness_
    sum_fitness_ = 0
    for i in range(0, shape_[0]):
        for j in range(0, shape_[1]): 
            sum_fitness_ = sum_fitness_ + compute_objective_function(i,j)*current_state_[i][j]
    #print("At iteration number:", count_, "The value of the fitness function:", sum_fitness_)
    #print("The optimal solution was found at:", best_global_)
    #print("List of optimal solutions obtained: ", personal_best_)

def summation_function():
    sum = 0
    for i in range(0, shape_[0]):
        for j in range(0, shape_[1]): 
            sum = sum + compute_objective_function(i,j)
    #print("The summation across the domain is:", sum)
    #np.sort(personal_best_)
    #print("List of optimal solutions obtained: ", personal_best_)
    
if __name__ == "__main__":
    shape_ = (100, 100)
    n = 4
    k = 2
    mean = np.zeros((n,2))
    covariance = np.zeros(n)
    for i in range(0,n):
        #print(i)
        mean[i][0] = np.random.randint(0, shape_[0])
        mean[i][1] = np.random.randint(0, shape_[1])
        covariance[i] = 1
        #print(mean[i])
    sum_of_gaussians = sum_gaussian.Sum_of_Gaussians(shape_,n,mean,covariance)
    terrain_map_ = sum_of_gaussians.gaussian_sum
    fig = plt.figure(figsize = (10,10))




    #print(np.max(terrain_map_))
    initialize_particles()
    speciation()
    #print(best_global_)


    for j in range(0, num_iter_):
        velocity_update()
        location_update()
        compute_fitness_function()
        count_ = count_ + 1
    summation_function()


    '''while (sum_fitness_ < 4.8): 
        velocity_update()
        location_update()
        compute_fitness_function()
        count_ = count_ + 1
    summation_function()'''


    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x = np.linspace(0, shape_[0] - 1, shape_[0])
    y = np.linspace(0, shape_[1] - 1, shape_[1])
    #print(x,y)
    X, Y = np.meshgrid(x,y)
    Z = terrain_map_
    ax.plot_wireframe(X,Y,Z, color = 'r', linewidth = 0.2)
    images = []
    image = ax.scatter3D([current_locations_[i][0] for i in range(n_particles_)],[current_locations_[i][1] for i in range(n_particles_)],[compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) for i in range(n_particles_)], c='b')
    images.append([image])
    animated_image = animation.ArtistAnimation(fig, images)
    animated_image.save('./pso3_output.gif', writer='pillow')
    im = Image.open('./pso3_output.gif')

