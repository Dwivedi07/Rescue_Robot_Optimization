#This is the first version of the PSO algorithm. It involves defining the inertia, social and cognition 
#weights manually. Is able to locate the local maxima quite well. 


import sum_gaussian
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

shape_ = (100, 100)
terrain_map_ = np.zeros(shape_)
n_species_ = 2
fraction_e_by_c_ = 2
collectors_num_ = 100
explorers_num_ = fraction_e_by_c_ * collectors_num_  
population_ = collectors_num_ + fraction_e_by_c_
current_state_ = np.zeros(shape_)
current_locations_ = np.zeros((population_, 2))
current_velocities_ = np.zeros((population_, 2))
maximum_velocity_ = shape_[0]
num_iter_ = 100
inertia_explorers_ = 0.01
social_explorers_ = 2
cognition_explorers_ = 1
inertia_collectors_ = 0.01
social_collectors_ = 2
cognition_collectors_ = 1
collectors_personal_optimal_values_ = np.zeros(collectors_num_)
explorers_personal_optimal_values_ = np.zeros(explorers_num_)
collectors_personal_best_locations_ = np.zeros((collectors_num_,2))
explorers_personal_best_locations_ = np.zeros((explorers_num_,2))
population_optimal_value_ = 0
population_best_location_ = np.zeros(2)
count_ = 0
sum_fitness_ = 0
vibration_ = 0.8 

def compute_objective_function(i,j):
    global terrain_map_ 
    int_i = int(i)
    int_j = int(j)
    return terrain_map_[int_i][int_j]

def initialize_particles():
    global shape_, terrain_map_, current_state_
    global current_locations_, current_velocities_
    global best_global_, global_optimal_value_
    global population_, vibration_
    global social_collectors_, cognition_collectors_, inertia_collectors_
    global social_explorers_, cognition_explorers_, inertia_explorers_
    global population_optimal_value_, population_best_location_
    for i in range(0, population_):
        if i < explorers_num_:
            current_locations_[i][0] = int(np.random.randint(0, shape_[0] - 1))
            current_locations_[i][1] = int(np.random.randint(0, shape_[1] - 1))
            current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 1
            explorers_personal_best_locations_[i] = current_locations_[i]
            explorers_personal_optimal_values_[i] = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
            if compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1])) > population_optimal_value_:
                population_optimal_value_ = compute_objective_function(int(current_locations_[i][0]), int(current_locations_[i][1]))
                population_best_location_ = current_locations_[i]
            current_velocities_[i][0] = int(np.random.randint(int(-shape_[0]*vibration_), int(shape_[0]*vibration_)))
            current_velocities_[i][1] = int(np.random.randint(int(-shape_[1]*vibration_), int(shape_[1]*vibration_)))



def velocity_update():
    global n_particles_, count_
    global current_velocities_, social_, cognition_, inertia_
    global personal_best_, best_global_, global_optimal_value_, personal_optimal_values_
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
    print("At iteration number:", count_, "The value of the fitness function:", sum_fitness_)
    print("The optimal solution was found at:", best_global_)
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
        print(mean[i])
    sum_of_gaussians = sum_gaussian.Sum_of_Gaussians(shape_,n,mean,covariance)
    terrain_map_ = sum_of_gaussians.gaussian_sum
    #fig = plt.figure(figsize = (10,10))

    #print(np.max(terrain_map_))
    initialize_particles()
    #print(best_global_)

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(0, shape_[0]), ylim=(0, shape_[1]))
    
    x = np.linspace(0, shape_[0] - 1, shape_[0])
    y = np.linspace(0, shape_[1] - 1, shape_[1])
    X, Y = np.meshgrid(x,y)
    Z = terrain_map_
    img = ax.imshow(Z, extent=[0, shape_[0] - 1, 0, shape_[1] - 1], origin='lower', cmap='Blues', alpha=0.5)
    particles, = ax.plot([], [], 'bo', ms=6)
    rect = plt.Rectangle((0,0),shape_[0], shape_[1], ec='none', lw=2, fc='none')
    ax.add_patch(rect)

    def init_animation():
        global current_locations_, current_velocities_, rect
        particles.set_data([], [])
        particles.set_color("red")
        rect.set_edgecolor('none')
        return particles, rect
    
    def animate(i):
        global count_, current_locations_, current_velocities_, terrain_map_
        velocity_update()
        location_update()
        compute_fitness_function()
        count_ = count_ + 1
        ms = int(fig.dpi * 2 * 0.2 * fig.get_figwidth()/ np.diff(ax.get_xbound())[0])
        rect.set_edgecolor('k')
        particles.set_data(current_locations_[:,1], current_locations_[:,0])
        particles.set_markersize(ms)
        return particles, rect
    
    ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=100, blit=True, init_func=init_animation)
    plt.show()


    # for j in range(0, num_iter_):
    #     velocity_update()
    #     location_update()
    #     compute_fitness_function()
    #     count_ = count_ + 1
    # summation_function()  


    # x = np.linspace(0, shape_[0] - 1, shape_[0])
    # y = np.linspace(0, shape_[1] - 1, shape_[1])
    # X, Y = np.meshgrid(x,y)
    # Z = terrain_map_
    # fig, ax = plt.subplots(figsize=(8,8))
    # img = ax.imshow(Z, extent=[0, shape_[0] - 1, 0, shape_[1] - 1], origin='lower', cmap='viridis', alpha=0.5)
    # p_plot = ax.scatter(current_locations_[:,1], current_locations_[:,0], marker='o', color='blue', alpha=0.5)
    # plt.show()
    # fig = plt.figure()
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
    #                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))
    # particles, = ax.plot([], [], 'bo', ms=6)
    # rect = plt.Rectangle((0,0),shape_[0], shape_[1], ec='none', lw=2, fc='none')
    # ax.add_patch(rect)