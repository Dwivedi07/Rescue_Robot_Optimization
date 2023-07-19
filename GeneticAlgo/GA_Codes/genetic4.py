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

#Define the sensor related details
total_num_sensors_= 20 #Defines the total number of sensors
types_of_sensors_ = 1   #Defines the types of sensors 
sensor_type_listing_ = [20]
sensor_radii_listing_ = [10]


shape_ = (100, 100)
num_iter =   100
n_samples_ = 100
iter_count_ = 0

fittest_member_ = 0

sensor_array_ = np.zeros((total_num_sensors_, 3))
population_ = 80
selection_to_next_generation_ = population_ 


mutation_std_dev_ = shape_[0]/2
mutation_rate_ = 0.016
mutation_update_ = 0
mutation_factor_ = 2
success_mutation_counter_ = 0
failure_mutation_counter_ = 0 
type_mutation = "basic" #Choose "basic" or "better"


selection_exponent_ = 20
selection_exponent_update_ = 10

#half the mutation rate as specified by the paper, since we are using a different 
#mutation scheme 
strings_population_ = np.zeros((population_, total_num_sensors_, 3))
best_configuration_ = np.zeros((num_iter, total_num_sensors_, 3))
fitness_population_ = np.zeros(population_)
store_average_fitness_ = np.zeros(num_iter)
iter_list_ = []
avg_fit_list_ = []
max_fit_list_ = []
best_iter_ = 0
best_organism_ = 0 
best_fitness_value_ = 0 
if(type_mutation == "basic"): 
    name_string_ = "Pop_" + str(population_)  + "_mtype_" + type_mutation + "_mut_" + str(mutation_rate_) + "_mup_" + str(mutation_update_) +  "_nsamples_" + str(n_samples_) + "_niter_" + str(num_iter) + "_sexp_" + str(selection_exponent_) + "_seup_" + str(selection_exponent_update_)
if(type_mutation == "better"): 
    name_string_ = "Pop_" + str(population_)  + "_mtype_" + type_mutation + "_MutFact_" + str(mutation_factor_) +  "_nsamples_" + str(n_samples_) + "_niter_" + str(num_iter) + "_sexp_" + str(selection_exponent_) + "_seup_" + str(selection_exponent_update_)
#strings_population stores the complete population 
#strings_population[i] gives the i^th member of the population 
#strings_population[i][j][0] : radius of j^th sensor in i^th population 
#strings_population[i][j][1] : abscissa of center of j^th sensor in i^th population 
#strings_population[i][j][2] : ordinate of center of j^th sensor in i^th population 
#print(strings_population_)


#Define the domain over which the optimization is supposed to be done and create a union of polygons
listpoly_ = []
polygon_1_ = Polygon([(2,2), (2,98), (49,98), (49,2)])
listpoly_.append(polygon_1_)


listobs_ = []

def define_sensors(): 
    global total_num_sensors_, types_of_sensors_, sensor_type_listing_, sensor_radii_listing_, strings_population_
    global population_, strings_population_
    for i in range(0, population_): 
        sum_old = 0 
        sum_new = 0
        for j in range(0, types_of_sensors_): 
            sum_new = sum_old + sensor_type_listing_[j]
            for k in range(sum_old, sum_new): 
                strings_population_[i][k][0] = sensor_radii_listing_[j]
            sum_old = sum_new

def is_inside_union_of_polygons(listpoly, test_point): 
    inside_check = False
    for lst in listpoly: 
        if(test_point.within(lst) == True): 
            inside_check = True
    return inside_check
    
def is_not_hitting_obstacle(listobs, test_point): 
    hit_check = True
    for lst in listobs:
        if(test_point.within(lst) == True):
            hit_check = False
    return hit_check

def calculate_area(lista): 
    sum = 0
    for lst in lista: 
        sum = sum + lst.area
    return sum 

def initialize_population():
    global population_, strings_population_, shape_, total_num_sensors_
    for i in range(0, population_):
        for j in range(0, total_num_sensors_):
            strings_population_[i][j][1] = int(np.random.randint(0,shape_[0] - 1))
            strings_population_[i][j][2] = int(np.random.randint(0,shape_[1] - 1))

def is_point_under_coverage(x,y,i):
    global total_num_sensors_, strings_population_
    yes_it_is = False
    yes_it_is = 0 
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

def sensor_detection(): 
    return 1 

#The results obtained with 10^3, 10^4 and 10^5 are quite close to each other, so
#to reduce execution time (0.3 seconds per string), select number_of_random_samples = 1000 
def monte_carlo(i):
    global shape_, total_num_sensors_, strings_population_, population_, n_samples_
    global listobs_, listpoly_
    area_domain = calculate_area(listpoly_) - calculate_area(listobs_)
    count = 0 
    samples_inside_polygon = 1 
    for j in range(0, n_samples_):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage(x,y,i)
        trial_point = Point(x,y)
        is_good_ = is_inside_union_of_polygons(listpoly_, trial_point) and is_not_hitting_obstacle(listobs_, trial_point)
        if(is_good_ == True):
            samples_inside_polygon = samples_inside_polygon + 1
            if (temp_cover): 
                count = count + 1
    estimated_fraction = count/samples_inside_polygon
    #efficiency = (count*area_max_possible_)/(n_samples_*area)
    estimated_area = estimated_fraction*area_domain
    #print("The estimated area is:", estimated_area)
    return estimated_fraction

def monte_carlo_2(input_string, i):
    global shape_, total_num_sensors_, strings_population_, population_, n_samples_, polygon_a_ , polygon_b_
    global listobs_, listpoly_
    area_domain = calculate_area(listpoly_) - calculate_area(listobs_)
    count = 0 
    samples_inside_polygon = 1 
    for j in range(0, n_samples_):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage_2(input_string,x,y,i)
        trial_point = Point(x,y)
        is_good_ = is_inside_union_of_polygons(listpoly_, trial_point) and is_not_hitting_obstacle(listobs_, trial_point)
        if(is_good_ == True):
            samples_inside_polygon = samples_inside_polygon + 1
            if (temp_cover): 
                count = count + 1
    estimated_fraction = count/samples_inside_polygon
    estimated_area = estimated_fraction*area_domain
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
    selection_probabilities = np.zeros(population_)
    for j in range(0, population_):
        selection_probabilities[j] = func_fitness[j]/np.sum(func_fitness)
    while (count < selection_to_next_generation_): 
        a = np.random.randint(0, population_)
        x = np.random.uniform(0,1)
        if x < selection_probabilities[int(a)]:
            temp_strings_population_[count] = strings_population_[int(a)]
            count = count + 1
    strings_population_ = temp_strings_population_
    fitness_new_population = np.zeros(selection_to_next_generation_)
    for i in range(0, selection_to_next_generation_): 
        fitness_new_population[i] = monte_carlo(i)
    store_average_fitness_[iter_count_] = np.sum(fitness_new_population)/selection_to_next_generation_
    iter_count_ = iter_count_ + 1
    print("Average fitness of population in iteration", iter_count_, 'is', np.sum(fitness_new_population)/selection_to_next_generation_)
    print("Max fitness of population in iteration", iter_count_, 'is', np.max(fitness_new_population)  )
    avg_fit_list_.append(store_average_fitness_[iter_count_ - 1])
    max_fit_list_.append(np.max(fitness_new_population))
    fittest_member_ = np.argmax(fitness_new_population) 
    best_configuration_[iter_count_ - 1] = strings_population_[fittest_member_]
    best_iter_ = iter_count_ - 1 
    best_fitness_value_ = np.max(fitness_new_population)


#We shall be using a different mutation operator than what is mentioned in the paper
#What the paper does: Use Gaussian mutation
#What we do: completely change the x-component or y-component for a sensor  
#through a uniform distribution 
def basic_mutation():
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


#This is ideally a better mutation function because it checks whether the newly created individual is better
#before actually applying the mutation
def better_mutation(): 
    global success_mutation_counter_, failure_mutation_counter_
    global strings_population_, shape_, population_
    global mutation_factor_, total_num_sensors_
    for i in range(0, population_): #Do for every organism in the population
        temp_string = strings_population_.copy()
        for j in range(0, mutation_factor_): #How many mutations in every organism? 
            k = int(np.random.randint(0,total_num_sensors_))
            l = int(np.random.randint(1,3))
            temp_string[i][k][l] = int(np.random.randint(0, shape_[1]-1 ))
        #print("Sensor k", k, "was mutated at position", l)
        #print("TS:", temp_string[i][k][l])
        #print("SP:", strings_population_[i][k][l])
        fitness_original = monte_carlo(i)
        fitness_mutated = monte_carlo_2(temp_string, i)
        if(fitness_mutated >= fitness_original):
            #print("For organism ", i, " Mutation successful", fitness_mutated, "is more than", fitness_original)
            #print("Old SP was", strings_population_[i])
            strings_population_ = temp_string.copy() 
            #print("New SP is", strings_population_[i])
            success_mutation_counter_ += 1
        else:
            #print("Did not mutate")
            failure_mutation_counter_ += 1
    ratio = success_mutation_counter_/(success_mutation_counter_ + failure_mutation_counter_)
    print("Rate of successful mutations is:", ratio )





#This function incorporates a clock-based mutation scheme as proposed by the paper 
#"Analyzing Mutation Schemes for Real-Parameter Genetic Algorithms" by Kalyanmoy and Debayan Deb 
def clock_mutation():
    global population_, strings_population_, shape_
    global mutation_rate_, mutation_std_dev_, total_num_sensors_

def update_rates():
    global num_iter, mutation_rate_, mutation_update_, selection_exponent_, selection_exponent_update_
    mutation_rate_ = mutation_rate_ + mutation_update_/num_iter
    selection_exponent_ = selection_exponent_ + selection_exponent_update_/num_iter
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

def edit_list_of_polygons(): 
    global listpoly_
    listpoly_.pop()
    polygon_2 = Polygon([(51,2), (51,98), (98, 98), (98, 2)])
    listpoly_.append(polygon_2)

def edit_list_of_obstacles():
    global listobs_
        
if __name__ == "__main__":
    start_time = time.time()
    print("Starting iteration:" + str(name_string_))
    #initialize_radii()
    define_sensors()
    initialize_population()
    circles_sensors_ = [0]*total_num_sensors_

    for k in range (0, num_iter):
        iter_list_.append(k)
        selection()
        weak_crossover()
        basic_mutation()
        update_rates()
        end_time = time.time()
        print("Cumulative execution time till iteration", iter_count_," = ", end_time - start_time )
        if(k == int(num_iter/10 - 1)): 
            edit_list_of_polygons()
        print(listpoly_)
        if (k % 5 == 0): 
            fig, ax = plt.subplots()
            plt.ion()
            for i in range(0, total_num_sensors_):
                circles_sensors_[i] = plt.Circle((best_configuration_[k][i][1], best_configuration_[k][i][2]), best_configuration_[k][i][0],  fill = True,  linewidth = 5)
                ax.add_patch(circles_sensors_[i]), 
            ax.set_xlim(xmin = 0, xmax= shape_[0])
            ax.set_ylim(ymin = 0, ymax = shape_[1])
            plt.xlabel(name_string_)
            est_max = monte_carlo_2(best_configuration_, k)
            plt.ylabel("At iteration " + str(k) + "Estimated fraction of area covered is: " + str(best_fitness_value_))
            for lst in listpoly_:
                plt.plot(*lst.exterior.xy, color = 'red')
            for lst in listobs_: 
                plt.plot(*lst.exterior.xy, color = 'blue')
            plt.draw()
            plt.pause(5)
            plt.close()

    #print("The maximum fitness found out is: ", max_num_in_list(max_fit_list_))
    #print("The radii of sensors are", sensor_array_[:] )






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


