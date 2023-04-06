import sum_gaussian
import numpy as np

class PSO: 
    
    def __init__(self, shape, n_particles, terrain_map, num_iter): 
        self.shape = shape
        self.terrain_map = terrain_map
        self.n_particles = n_particles
        self.sensor_locations = np.zeros((n_particles , 2))
        self.current_state = np.zeros(shape)
        self.current_locations = np.zeros((n_particles, 2))
        self.current_velocities = np.zeros((n_particles, 2))
        self.best_global = np.zeros(2)
        self.optimal_value = 0
        self.personal_optimal_values = np.zeros(n_particles)
        self.best_personal = np.zeros((n_particles , 2))
        self.num_iter = num_iter
        self.vibration = 0.1
        self.inertia = 1
        self.social = 0
        self.cognition = 0
        self.count = 0
        self.main_loop = self.main_loop()
    

    def compute_objective_function(self, i, j):
        return self.terrain_map[i][j]


    def initialize_particles(self): 
        for i in range(0, self.n_particles): 
            x = np.random.randint(0, self.shape[0] - 1)
            y = np.random.randint(0, self.shape[1] - 1)
            self.current_locations[i][0] = int(x)
            self.current_locations[i][1] = int(y)
            #print(self.current_locations[i][0], self.current_locations[i][1])
            self.current_state[int(self.current_locations[i][0])][int(self.current_locations[i][1])] = 1
            #print(self.current_state[int(self.current_locations[i][0])][int(self.current_locations[i][1])])
            self.best_personal[i][0] = x
            self.best_personal[i][1] = y
            #print(self.best_personal[i])
            self.personal_optimal_values[i] = self.compute_objective_function(x,y)
            #print(self.personal_optimal_values[i])
            if self.compute_objective_function(x,y) > self.optimal_value: 
                self.optimal_value = self.compute_objective_function(x,y)
                self.best_global[0] = x
                self.best_global[1] = y
            #print(self.best_global)
            self.current_velocities[i][0] = np.random.randint(int(-self.shape[0]*self.vibration), int(self.shape[1]*self.vibration))
            self.current_velocities[i][1] = np.random.randint(int(-self.shape[0]*self.vibration), int(self.shape[1]*self.vibration))
            #print(i, self.current_velocities[i])

    def velocity_update(self): 
        for i in range(0, self.n_particles):
            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)
            self.current_velocities[i][0] = int(self.inertia * self.current_velocities[i][0] + r1*self.cognition*(self.best_personal[i][0] - self.current_locations[i][0]) + r2*self.social*(self.best_global[0] - self.current_locations[i][0]))
            self.current_velocities[i][1] = int(self.inertia * self.current_velocities[i][1] + r1*self.cognition*(self.best_personal[i][1] - self.current_locations[i][1]) + r2*self.social*(self.best_global[1] - self.current_locations[i][1]))
            #print(i, r1, r2, self.current_velocities[i])

    def location_update(self): 
        for i in range(0, self.n_particles):
            self.current_locations[i][0] = int(self.current_locations[i][0])
            self.current_locations[i][1] = int(self.current_locations[i][1])
            self.current_state[int(self.current_locations[i][0])][int(self.current_locations[i][1])] = 0
            if self.current_locations[i][0] + self.current_velocities[i][0] <= self.shape[0]:
                if self.current_locations[i][0] + self.current_velocities[i][0] > 0:
                    self.current_locations[i][0] = self.current_locations[i][0] + self.current_velocities[i][0]
                self.current_locations[i][0] = 0
            self.current_locations[i][0] = self.shape[0] - 1
            if self.current_locations[i][1] + self.current_velocities[i][1] <= self.shape[1]:
                if self.current_locations[i][1] + self.current_velocities[i][1] > 0:
                    self.current_locations[i][1] = self.current_locations[i][1] + self.current_velocities[i][1]
                self.current_locations[i][1] = 0
            self.current_locations[i][1] = self.shape[1] - 1 
            self.current_state[int(self.current_locations[i][0])][int(self.current_locations[i][1])] = 1
            if self.compute_objective_function(int(self.current_locations[i][0]),int(self.current_locations[i][1])) > self.optimal_value: 
                self.optimal_value = self.compute_objective_function(self.current_locations[i][0],self.current_locations[i][1])
                self.best_global[0] = self.current_locations[i][0]
                self.best_global[1] = self.current_locations[i][1]
            if self.compute_objective_function(int(self.current_locations[i][0]),int(self.current_locations[i][1])) > self.personal_optimal_values[i]:
                self.personal_optimal_values[i] = self.compute_objective_function(int(self.current_locations[i][0]),int(self.current_locations[i][1]))
                self.best_personal[i][0] = self.current_locations[i][0]
                self.best_personal[i][1] = self.current_locations[i][1]
    
    def print_values(self):
        global_sum = 0 
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1] ):
                global_sum = global_sum + self.compute_objective_function(i,j)*self.current_state[i][j]
        #print("Number of iterations completed: ", self.count)
        #print("The value of the fitness function is: ", global_sum)
        #print("Location of first particle", self.current_locations[0])
        #print("Current state", self.current_state)
    
    def main_loop(self):
        self.initialize_particles()
        for l in range(0, self.num_iter):
            self.location_update()
            self.velocity_update()
            self.print_values()
            self.count = self.count  +  1 



                     
if __name__ == '__main__':
    a = (100, 100)
    n = 4
    k = 2
    mean = np.zeros((n,n))
    covariance = np.zeros(n)
    for i in range(0,n): 
        mean[i][0] = np.random.randint(1, a[0])
        mean[i][1] = np.random.randint(1, a[0])
        covariance[i] = 5
    sum_of_gaussians = sum_gaussian.Sum_of_Gaussians(a,n,mean,covariance)
    pso = PSO(a, 100, sum_of_gaussians.gaussian_sum, 100)

            
    




