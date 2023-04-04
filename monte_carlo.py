import numpy as np 
import time
import matplotlib.pyplot as plt


shape_ = (100, 100)
total_num_sensors_ = 40
r1_num_sensors_ = 30
r2_num_sensors_ = 10
radius_1_ = 10
radius_2_ = 5
sensor_array_ = np.zeros((total_num_sensors_, 3))

def initialize_radii():
    global sensor_array_, total_num_sensors_, r1_num_sensors_, r2_num_sensors_
    global radius_1_, radius_2_
    for i in range(r1_num_sensors_):
        sensor_array_[i][0] = radius_1_
    for j in range(r1_num_sensors_, r2_num_sensors_):
        sensor_array_[i][0] = radius_2_

def initialize_locations():
    global total_num_sensors_, sensor_array_, shape_
    for i in range(0, total_num_sensors_): 
        sensor_array_[i][1] = int(np.random.randint(0, shape_[0] - 1))
        sensor_array_[i][2] = int(np.random.randint(0, shape_[1] - 1)) 

def is_point_under_coverage(x,y):
    global sensor_array_, total_num_sensors_
    yes_it_is = False
    for i in range(0, total_num_sensors_): 
        if np.sqrt((x - sensor_array_[i][1])**2 + (y - sensor_array_[i][2])**2) <= sensor_array_[i][0]:
            yes_it_is = True
            break
    return yes_it_is

def monte_carlo(number_of_random_samples):
    global shape_
    area = shape_[0]*shape_[1]
    count = 0 
    for i in range(0, number_of_random_samples):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage(x,y)
        if (temp_cover): 
            count = count + 1
    estimated_area = area*count/number_of_random_samples
    print("The estimated area is:", estimated_area)
    return estimated_area

if __name__ == "__main__":
    start_time = time.time()
    initialize_locations()
    initialize_radii()
    monte_carlo(10000)
    end_time = time.time()
    print("Time taken to execute is:", end_time - start_time)
    #print("The radii of sensors are", sensor_array_[:] )
    circles_sensors_ = [0]*total_num_sensors_
    fig, ax = plt.subplots()

    for i in range(0, r1_num_sensors_):
        circles_sensors_[i] = plt.Circle((sensor_array_[i][1], sensor_array_[i][2]), sensor_array_[i][0], color = 'b')
        ax.add_patch(circles_sensors_[i])
    for i in range(r1_num_sensors_, r2_num_sensors_):
        circles_sensors_[i] = plt.Circle((sensor_array_[i][1], sensor_array_[i][2]), sensor_array_[i][0], color = 'r')
        ax.add_patch(circles_sensors_[i])
    ax.set_xlim(xmin = 0, xmax= shape_[0])
    ax.set_ylim(ymin = 0, ymax = shape_[1])

    fig.savefig('plot_sensors.png')
    plt.show()