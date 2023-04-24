#For this implementation of the genetic algorithm we are referring to the paper 
#An efficient genetic algorithm for maximum coverage deployment in wireless sensor
#networks
#The authors of this paper are Yourim Yoon and Yong-Hyuk Kim 
import numpy as np 
import time

shape_ = (100, 100)
num_iter = 500
iter_count_ = 0
total_num_sensors_ = 40
r1_num_sensors_ = 5
n_samples_ = 100
r2_num_sensors_ = 35
radius_1_ = 20
radius_2_ = 5
sensor_array_ = np.zeros((total_num_sensors_, 3))
population_ = 100
selection_to_next_generation_ = 100 
mutation_std_dev_ = shape_[0]/2
mutation_rate_ = 1/total_num_sensors_ 
#half the mutation rate as specified by the paper, since we are using a different 
#mutation scheme 
strings_population_ = np.zeros((population_, total_num_sensors_, 3))
fitness_population_ = np.zeros(population_)
store_average_fitness_ = np.zeros(num_iter)
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
    global shape_, total_num_sensors_, strings_population_, population_, n_samples_
    area = shape_[0]*shape_[1]
    count = 0 
    for j in range(0, n_samples_):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage(x,y,i)
        if (temp_cover): 
            count = count + 1
    estimated_fraction = count/n_samples_
    estimated_area = estimated_fraction*area
    #print("The estimated area is:", estimated_area)
    return estimated_fraction

def monte_carlo_2(input_string, i):
    global shape_, total_num_sensors_, strings_population_, population_, n_samples_
    area = shape_[0]*shape_[1]
    count = 0 
    for j in range(0, n_samples_):
        x = np.random.uniform(0, shape_[0])
        y = np.random.uniform(0, shape_[1])
        temp_cover = is_point_under_coverage_2(input_string,x,y,i)
        if (temp_cover): 
            count = count + 1
    estimated_fraction = count/n_samples_
    estimated_area = estimated_fraction*area
    #print("The estimated area is:", estimated_area)
    return estimated_fraction




def selection(): 
    global population_, selection_to_next_generation_, strings_population_
    global fitness_population_, store_average_fitness_, iter_count_
    temp_strings_population_ = np.zeros((population_, total_num_sensors_, 3))
    iter_count_ = iter_count_ + 1
    count = 0
    func_fitness = np.zeros(population_)
    for i in range(0, population_): 
        fitness_population_[i] = monte_carlo(i)
        func_fitness[i] = fitness_population_[i]**20
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
    print("Average fitness of population in iteration", iter_count_, 'is', np.sum(fitness_new_population)/selection_to_next_generation_)
    print("Max fitness of population in iteration", iter_count_, 'is', np.max(fitness_new_population)  )
#We shall be using a different mutation operator than what is mentioned in the paper
#What the paper does: Use Gaussian mutation
#What we do: completely change the x-component or y-component for a sensor  
#through a uniform distribution 
def mutation():
    global population_, strings_population_, shape_
    global mutation_rate_, mutation_std_dev_, total_num_sensors_
    sigma = 10
    for i in range(0, population_):
        for j in range(0, total_num_sensors_):
            a = np.random.uniform(0,1)
            if a < mutation_rate_:
                strings_population_[i][j][1] = strings_population_[i][j][1] + int(np.random.normal(0, sigma)) # int(np.random.randint(0, shape_[0]-1 ))
            b = np.random.uniform(0,1)
            if b < mutation_rate_:
                strings_population_[i][j][2] = strings_population_[i][j][2] + int(np.random.normal(0, sigma)) #int(np.random.randint(0, shape_[1]-1 ))

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
        
if __name__ == "__main__":
    start_time = time.time()
    initialize_radii()
    initialize_population()
    for k in range (0, num_iter):
        selection()
        weak_crossover()
        mutation()
        end_time = time.time()
        print("Cumulative execution time till iteration", iter_count_," = ", end_time - start_time )
