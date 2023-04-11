#Implementation of PSO using 


import sum_gaussian
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

shape_ = (100, 100)
terrain_map_ = np.zeros(shape_)
n_particles_ = 100
solutions_ = []
current_state_ = np.zeros(shape_)
current_locations_ = np.zeros((n_particles_, 2))
current_velocities_ = np.zeros((n_particles_, 2))
maximum_velocity_ = shape_[0]
best_global_ = np.zeros(2)
global_optimal_value_ = 0
personal_best_ = np.zeros((n_particles_, 2))
personal_optimal_values_ = np.zeros(n_particles_)
num_iter_ = 50
inertia_ = np.zeros(n_particles_) + 0
social_ = 1
cognition_ = 1
count_ = 0
marker_ = np.zeros(n_particles_)
sum_fitness_ = 0
vibration_ = 1 
detected_humans_ = 0
gaussian_markers_ = np.zeros(shape_)
number_of_peaks_ = 3
locations_of_peaks_ = np.zeros((number_of_peaks_ , 2))
reach_count_ = np.zeros(number_of_peaks_)
confidence_factor_ = 0.8 

def compute_objective_function(i,j):
    global terrain_map_ 
    int_i = int(i)
    int_j = int(j)
    return terrain_map_[int_i][int_j]

def initialize_particles():
    global shape_, terrain_map_, current_state_, personal_best_
    global current_locations_, current_velocities_
    global best_global_, global_optimal_value_, personal_optimal_values_
    global n_particles_, vibration_
    for i in range(0, n_particles_):
        current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 0
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


def reinitialize_particles():
    global shape_, terrain_map_, current_state_, personal_best_
    global current_locations_, current_velocities_
    global best_global_, global_optimal_value_, personal_optimal_values_
    global n_particles_, vibration_
    for i in range(0, n_particles_):
        current_state_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 0
        current_locations_[i][0] = current_locations_[i][0] + int(np.random.randint(int(-shape_[0]/2), int(shape_[0]/2)))
        if current_locations_[i][0] > shape_[0] - 1 : 
            current_locations_[i][0] = shape_[0] - 1
        elif current_locations_[i][0] < -shape_[0] - 1:
            current_locations_[i][0] = -(shape_[0] - 1)
        current_locations_[i][1] = current_locations_[i][1] + int(np.random.randint(int(-shape_[1]/2), int(shape_[1]/2)))
        if current_locations_[i][1] > shape_[1] - 1 : 
            current_locations_[i][1] = shape_[1] - 1
        elif current_locations_[i][1] < -shape_[1] - 1:
            current_locations_[i][1] = -(shape_[1] - 1)
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

def reinitialize_velocities():
    global current_velocities_, shape_, vibration_
    for k in range(0, n_particles_):
        current_velocities_[k][0] = int(np.random.randint(int(-shape_[0]*vibration_), int(shape_[0]*vibration_)))
        current_velocities_[k][1] = int(np.random.randint(int(-shape_[1]*vibration_), int(shape_[1]*vibration_)))

def check_if_reached(i):
    global locations_of_peaks_, current_locations_, reach_count_, n_particles_, confidence_factor_
    reached = False
    proximity_count = 0 
    for j in range(number_of_peaks_):
        if np.sqrt((current_locations_[i][0] - locations_of_peaks_[j][0])**2 + (current_locations_[i][1] - locations_of_peaks_[j][1])**2) < 1.4:
            for k in range(n_particles_):
                if np.sqrt((locations_of_peaks_[j][0] - current_locations_[k][0])**2 + (locations_of_peaks_[j][1] - current_locations_[k][1])**2) < 5:
                    proximity_count += 1
            if proximity_count > confidence_factor_*n_particles_ :     
                reached = True
    return reached
    
def velocity_update():
    global n_particles_, count_
    global current_velocities_, social_, cognition_, inertia_
    global personal_best_, best_global_, global_optimal_value_, personal_optimal_values_, locations_of_peaks_
    for i in range(0, n_particles_):
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)
        #print(i, r1, r2)
        #print("Iteration number:", count_)
        #print(i, "The old velocity was:", current_velocities_[i])
        #print(i, "Social distance is:", best_global_ - current_locations_[i])
        current_velocities_[i][0] = int(inertia_[i] * current_velocities_[i][0] + r1*cognition_*(personal_best_[i][0] - current_locations_[i][0]) + r2*social_*(best_global_[0] - current_locations_[i][0]))
        current_velocities_[i][1] = int(inertia_[i] * current_velocities_[i][1] + r1*cognition_*(personal_best_[i][1] - current_locations_[i][1]) + r2*social_*(best_global_[1] - current_locations_[i][1]))
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
    global best_global_, personal_best_, detected_humans_
    global global_optimal_value_, personal_optimal_values_, marker_, solutions_

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
        # if current_velocities_[i][0] < 0.1 and current_velocities_[i][1] < 0.1 and current_velocities_[i][0] > 0  and current_velocities_[i][1] > 0  :
        #     gaussian_subtraction(current_locations_[i][0], current_locations_[i][1])
        #     current_velocities_[i][0] = 0
        #     current_velocities_[i][0] = 0 
        #     detected_humans_ = detected_humans_ + 1
        if check_if_reached(i): #and gaussian_markers_[int(current_locations_[i][0])][int(current_locations_[i][1])] == 0 :
            gaussian_subtraction(current_locations_[i][0], current_locations_[i][1])
            print("Gaussian subtracted at location ", current_locations_[i])
            #gaussian_markers_[int(current_locations_[i][0])][int(current_locations_[i][1])] = 1
            best_global_[0] = 0
            best_global_[1] = 0
            global_optimal_value_ = 0
            for i in range(n_particles_):
                personal_best_[i][0] = 0
                personal_best_[i][1] = 0
                personal_optimal_values_[i] = 0
            initialize_particles()



def gaussian_subtraction(x,y):
    global terrain_map_
    cov = 1
    for j in range(0,shape_[0]):
        for k in range(0, shape_[1]): 
            terrain_map_[j][k] = terrain_map_[j][k] - 2*np.exp(-0.5*((j - x)**2 + (k - y)**2)/(cov**2)) 

        

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
    covariance = np.zeros(number_of_peaks_)
    for i in range(0,number_of_peaks_):
        #print(i)
        locations_of_peaks_[i][0] = np.random.randint(0, shape_[0])
        locations_of_peaks_[i][1] = np.random.randint(0, shape_[1])
        covariance[i] = 1
        print(locations_of_peaks_[i])
    sum_of_gaussians = sum_gaussian.Sum_of_Gaussians(shape_,number_of_peaks_,locations_of_peaks_,covariance)
    terrain_map_ = sum_of_gaussians.gaussian_sum
    #fig = plt.figure(figsize = (10,10))

    # x = np.linspace(0, shape_[0] - 1, shape_[0])
    # y = np.linspace(0, shape_[1] - 1, shape_[1])
    # X, Y = np.meshgrid(x,y)
    # Z = terrain_map_
    # fig, bx = plt.subplots(figsize=(8,8))
    # img = bx.imshow(Z, extent=[0, shape_[0] - 1, 0, shape_[1] - 1], origin='lower', cmap='viridis', alpha=0.5)



    initialize_particles()
    #speciation()
    fig = plt.figure()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(0, shape_[0]), ylim=(0, shape_[1]))
    
    x = np.linspace(0, shape_[0] - 1, shape_[0])
    y = np.linspace(0, shape_[1] - 1, shape_[1])
    X, Y = np.meshgrid(x,y)
    Z = terrain_map_
    img = ax.imshow(Z, extent=[0, shape_[0] - 1, 0, shape_[1] - 1], origin='lower', cmap='YlGn', alpha=0.5)
    particles, = ax.plot([], [], 'bo', ms=6)
    rect = plt.Rectangle((0,0),shape_[0], shape_[1], ec='none', lw=2, fc='none')
    ax.add_patch(rect)

    def init_animation():
        global current_locations_, current_velocities_, rect
        particles.set_data([], [])
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
                              interval=200, blit=True, init_func=init_animation)
    plt.show()
