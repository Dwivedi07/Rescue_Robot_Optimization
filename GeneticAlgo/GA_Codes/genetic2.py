#For this implementation of the genetic algorithm we are referring to the paper 
#An efficient genetic algorithm for maximum coverage deployment in wireless sensor
#networks
#The authors of this paper are Yourim Yoon and Yong-Hyuk Kim 
import numpy as np 
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Point, Polygon, MultiPolygon
from wktplot import WKTPlot
from skimage import io, measure, morphology
from skimage.io import imsave, imread

plt.style.use('dark_background')
polygon_a_ = Polygon([(5,97), (37,97), (53,84), (29,57), (4,71)])
polygon_b_ = Polygon([(29,42), (62,59), (84,35), (64,6), (32,14)])
shape_ = (100, 100)
num_iter = 50
n_samples_ = 500
iter_count_ = 0
total_num_sensors_ = 15
r1_num_sensors_ = 5
r2_num_sensors_ = 10
fittest_member_ = 0
radius_1_ = 10
radius_2_ = 5
area_max_possible_ = np.pi*(r1_num_sensors_*radius_1_**2 + r2_num_sensors_*radius_2_**2)
sensor_array_ = np.zeros((total_num_sensors_, 3))
population_ = 80
selection_to_next_generation_ = population_ 
mutation_std_dev_ = shape_[0]/2
mutation_rate_ = 0.02
selection_exponent_ = 20 
#half the mutation rate as specified by the paper, since we are using a different 
#mutation scheme 
strings_population_ = np.zeros((population_, total_num_sensors_, 3))
best_configuration_ = np.zeros((1, total_num_sensors_, 3))
fitness_population_ = np.zeros(population_)
store_average_fitness_ = np.zeros(num_iter)
iter_list_ = []
avg_fit_list_ = []
max_fit_list_ = []
best_iter_ = 0
best_organism_ = 0 
best_fitness_value_ = 0 
name_string_ = "Pop_" + str(population_) + "_n1_" + str(r1_num_sensors_) + "_r1_" + str(radius_1_) + "_n2_" + str(r2_num_sensors_) + "_r2_" + str(radius_2_) + "_mut_" + str(mutation_rate_) + "_nsamples_" + str(n_samples_) + "_niter_" + str(num_iter) + "_sexp_" + str(selection_exponent_)
#strings_population stores the complete population 
#strings_population[i] gives the i^th member of the population 
#strings_population[i][j][0] : radius of j^th sensor in i^th population 
#strings_population[i][j][1] : abscissa of center of j^th sensor in i^th population 
#strings_population[i][j][2] : ordinate of center of j^th sensor in i^th population 
#print(strings_population_)


def initialize_radii():
    global population_, r1_num_sensors_, radius_1_, radius_2_
    global strings_population_
    for i in range(0, population_):     
        for j in range(0, r1_num_sensors_): 
            strings_population_[i][j][0] = radius_1_
            #print(strings_population_[i][j][0])
    for i in range(0, population_):     
        for j in range(r1_num_sensors_, total_num_sensors_): 
            strings_population_[i][j][0] = radius_2_
            #print(strings_population_[i][j][0])

def initialize_population():
    global population_, strings_population_, shape_, total_num_sensors_
    for i in range(0, population_):
        for j in range(0, total_num_sensors_):
            strings_population_[i][j][1] = int(np.random.randint(0,shape_[0] - 1))
            strings_population_[i][j][2] = int(np.random.randint(0,shape_[1] - 1))

def is_point_under_coverage(x,y,i):
    global total_num_sensors_, strings_population_
    yes_it_is = False
    for j in range(0, total_num_sensors_): 
        if np.sqrt((x - strings_population_[i][j][1])**2 + (y - strings_population_[i][j][2])**2) <= strings_population_[i][j][0]:
            yes_it_is = True
            break
    return yes_it_is

def is_point_under_coverage_2(input_string, x,y,i):
    global total_num_sensors_
    yes_it_is = False
    for j in range(0, total_num_sensors_): 
        if np.sqrt((x - input_string[i][j][1])**2 + (y - input_string[i][j][2])**2) <= input_string[i][j][0]:
            yes_it_is = True
            break
    return yes_it_is     

#The results obtained with 10^3, 10^4 and 10^5 are quite close to each other, so
#to reduce execution time (0.3 seconds per string), select number_of_random_samples = 1000 
def monte_carlo(i):
    global shape_, total_num_sensors_, strings_population_, population_, n_samples_, polygon_a_, polygon_b_
    area = shape_[0]*shape_[1]
    area_polygon = (polygon_a_.area + polygon_b_.area) 
    count = 0 
    samples_inside_polygon = 1 
    for j in range(0, n_samples_):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage(x,y,i)
        trial_point = Point(x,y)
        is_within_polygon_ = trial_point.within(polygon_a_) + trial_point.within(polygon_b_)
        if(is_within_polygon_):
            samples_inside_polygon = samples_inside_polygon + 1
            if (temp_cover): 
                count = count + 1
    estimated_fraction = count/samples_inside_polygon
    #efficiency = (count*area_max_possible_)/(n_samples_*area)
    estimated_area = estimated_fraction*area_polygon
    #print("The estimated area is:", estimated_area)
    return estimated_fraction

def monte_carlo_2(input_string, i):
    global shape_, total_num_sensors_, strings_population_, population_, n_samples_, polygon_a_ , polygon_b_
    area = shape_[0]*shape_[1]
    area_polygon = (polygon_a_.area + polygon_b_.area) 
    count = 0 
    samples_inside_polygon = 1 
    for j in range(0, n_samples_):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage_2(input_string,x,y,i)
        trial_point = Point(x,y)
        is_within_polygon_ = trial_point.within(polygon_a_) or trial_point.within(polygon_b_)
        if(is_within_polygon_):
            samples_inside_polygon = samples_inside_polygon + 1
            if (temp_cover): 
                count = count + 1
    estimated_fraction = count/samples_inside_polygon
    estimated_area = estimated_fraction*area_polygon
    #print("The estimated area is:", estimated_area)
    return estimated_fraction




def selection(): 
    global population_, selection_to_next_generation_, strings_population_, max_fit_list_, iter_list_, avg_fit_list_
    global fitness_population_, store_average_fitness_, iter_count_, fittest_member_, best_iter_, best_fitness_value_, best_configuration_
    global selection_exponent_
    temp_strings_population_ = np.zeros((population_, total_num_sensors_, 3))
    count = 0
    func_fitness = np.zeros(population_)
    for i in range(0, population_): 
        fitness_population_[i] = monte_carlo(i)
        func_fitness[i] = fitness_population_[i]**selection_exponent_
        #print("Fitness of member" ,i, "in original population is:", fitness_population_[i])
    #print("Average fitness of population = ", np.sum(fitness_population_)/population_)
    selection_probabilities = np.zeros(population_)
    for j in range(0, population_):
        #selection_probabilities[i] = fitness_population_[i]/np.sum(fitness_population_)
        selection_probabilities[j] = func_fitness[j]/np.sum(func_fitness)
    #print(fitness_population_)
    #print(selection_probabilities)
    while (count < selection_to_next_generation_): 
        a = np.random.randint(0, population_)
        x = np.random.uniform(0,1)
        if x < selection_probabilities[int(a)]:
            temp_strings_population_[count] = strings_population_[int(a)]
            count = count + 1
            #print("Selected member ", int(a))
    strings_population_ = temp_strings_population_
    #print(strings_population_)
    fitness_new_population = np.zeros(selection_to_next_generation_)
    for i in range(0, selection_to_next_generation_): 
        fitness_new_population[i] = monte_carlo(i)
        #print("Fitness of member", i, "in new population is:", fitness_new_population[i])
    store_average_fitness_[iter_count_] = np.sum(fitness_new_population)/selection_to_next_generation_
    iter_count_ = iter_count_ + 1
    print("Average fitness of population in iteration", iter_count_, 'is', np.sum(fitness_new_population)/selection_to_next_generation_)
    print("Max fitness of population in iteration", iter_count_, 'is', np.max(fitness_new_population)  )
    avg_fit_list_.append(store_average_fitness_[iter_count_ - 1])
    max_fit_list_.append(np.max(fitness_new_population))
    if (np.max(fitness_new_population) >= max_num_in_list(max_fit_list_)):
        fittest_member_ = np.argmax(fitness_new_population) 
        best_configuration_[0] = strings_population_[fittest_member_]
        best_iter_ = iter_count_ - 1 
        best_fitness_value_ = max_num_in_list(max_fit_list_)


#We shall be using a different mutation operator than what is mentioned in the paper
#What the paper does: Use Gaussian mutation
#What we do: completely change the x-component or y-component for a sensor  
#through a uniform distribution 
def mutation():
    global population_, strings_population_, shape_
    global mutation_rate_, mutation_std_dev_, total_num_sensors_
    for i in range(0, population_):
        for j in range(0, total_num_sensors_):
            a = np.random.uniform(0,1)
            if a < mutation_rate_:
                strings_population_[i][j][1] = int(np.random.randint(0, shape_[0]-1 ))
            b = np.random.uniform(0,1)
            if b < mutation_rate_:
                strings_population_[i][j][2] = int(np.random.randint(0, shape_[1]-1 ))

#This function incorporates a clock-based mutation scheme as proposed by the paper 
#"Analyzing Mutation Schemes for Real-Parameter Genetic Algorithms" by Kalyanmoy and Debayan Deb 
def clock_mutation():
    global population_, strings_population_, shape_
    global mutation_rate_, mutation_std_dev_, total_num_sensors_

#We are not using the crossover operator which the paper suggests (BLX-\alpha) operator
#Instead we are using a more generic and weak crossover operator where we break the
# string into two parts and just merge the two strings thus generated 
#However, we are trying to include conditions which will try to drive the system 
#towards members which have greater fitness values
def weak_crossover():
    global population_, strings_population_, total_num_sensors_
    offspring_temp = np.zeros((population_, total_num_sensors_, 3))
    a = np.random.randint(1, total_num_sensors_ - 1)
    for i in range(0, int(population_/2)):
        for j in range(0, a):
            offspring_temp[2*i][j] = strings_population_[2*i][j]
            offspring_temp[2*i + 1][j] = strings_population_[2*i + 1][j]
        for j in range(a, total_num_sensors_):
            offspring_temp[2*i][j] = strings_population_[2*i + 1][j] 
            offspring_temp[2*i + 1][j] = strings_population_[2*i][j]
        f_old_2i = monte_carlo(2*i)
        f_old_2i_plus_1 = monte_carlo(2*i + 1)
        f_new_2i = monte_carlo_2(offspring_temp, 2*i )
        f_new_2i_plus_1 = monte_carlo_2(offspring_temp, 2*i + 1)
        if f_old_2i < f_new_2i:
            strings_population_[2*i] = offspring_temp[2*i]
        if f_old_2i_plus_1 < f_new_2i_plus_1:
            strings_population_[2*i + 1] = offspring_temp[2*i + 1]

def max_num_in_list(list):
    max = list[ 0 ]
    for a in list:
        if a > max:
            max = a
    return max
        
if __name__ == "__main__":
    start_time = time.time()
    print("Starting iteration:" + str(name_string_))
    initialize_radii()
    initialize_population()
    for k in range (0, num_iter):
        iter_list_.append(k)
        selection()
        weak_crossover()
        mutation()
        end_time = time.time()
        print("Cumulative execution time till iteration", iter_count_," = ", end_time - start_time )

    #print("The maximum fitness found out is: ", max_num_in_list(max_fit_list_))
    #print("The radii of sensors are", sensor_array_[:] )
    circles_sensors_ = [0]*total_num_sensors_
    fig, ax = plt.subplots()

    for i in range(0, total_num_sensors_):
        if i < r1_num_sensors_:
            circles_sensors_[i] = plt.Circle((best_configuration_[0][i][1], best_configuration_[0][i][2]), best_configuration_[0][i][0], fill = True,  linewidth = 5)
            #circles_sensors_[i] = plt.Circle((strings_population_[fittest_member_][i][1], strings_population_[fittest_member_][i][2]), strings_population_[fittest_member_][i][0], fill = True,  linewidth = 5)
        else:
            circles_sensors_[i] = plt.Circle((best_configuration_[0][i][1], best_configuration_[0][i][2]), best_configuration_[0][i][0],  fill = True,  linewidth = 5)
            #circles_sensors_[i] = plt.Circle((strings_population_[fittest_member_][i][1], strings_population_[fittest_member_][i][2]), strings_population_[fittest_member_][i][0],  fill = True,  linewidth = 5)
        ax.add_patch(circles_sensors_[i]), 
    ax.set_xlim(xmin = 0, xmax= shape_[0])
    ax.set_ylim(ymin = 0, ymax = shape_[1])
    plt.xlabel(name_string_)
    est_max = monte_carlo_2(best_configuration_, 0)
    plt.ylabel("Estimated fraction of area covered is: " + str(best_fitness_value_))
    plt.plot(*polygon_a_.exterior.xy)
    plt.plot(*polygon_b_.exterior.xy)
    fig.savefig('genetic_output.png')
    #print("List of average fitness values")
    #print(avg_fit_list_)
    #print("List of maximum fitness values")
    #print(max_fit_list_)
    print(name_string_)
    print("The maximum fitness found out is: ", max_num_in_list(max_fit_list_))
    print("The maximum average fitness found out is: ", max_num_in_list(avg_fit_list_))
    line_array = np.zeros(num_iter)

    fig2, ax2 = plt.subplots()
    max_pop_plot_x = np.array(iter_list_)
    max_pop_plot_y = np.array(max_fit_list_)
    avg_pop_plot_y = np.array(avg_fit_list_)
    plt.plot(max_pop_plot_x, max_pop_plot_y)
    plt.plot(max_pop_plot_x, avg_pop_plot_y)
    plt.xlabel(name_string_)
    plt.show()
    # fig = plt.figure()
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
    #                     xlim=(0, shape_[0]), ylim=(0, shape_[1]))
    # particles, = ax.plot([], [], 'bo', ms=6)
    # rect = plt.Rectangle((0,0),shape_[0], shape_[1], ec='none', lw=2, fc='none')
    # ax.add_patch(rect)

    # def init_animation():
    #     global strings_population_, rect
    #     particles.set_data([], [])
    #     particles.set_color("red")
    #     rect.set_edgecolor('none')
    #     return particles, rect
    
    # def animate(i):
    #     global count_, current_locations_, current_velocities_, terrain_map_
    #     selection()
    #     weak_crossover()
    #     mutation()
    #     count_ = count_ + 1
    #     ms = int(fig.dpi * 2 * 0.2 * fig.get_figwidth()/ np.diff(ax.get_xbound())[0])
    #     rect.set_edgecolor('k')
    #     particles.set_data(current_locations_[:,1], current_locations_[:,0])
    #     particles.set_markersize(ms)
    #     return particles, rect
    
    # ani = animation.FuncAnimation(fig, animate, frames=600,
    #                           interval=100, blit=True, init_func=init_animation)
    # plt.show()