import numpy as np
import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
from Environment_generate import generate_features

class Sampling():
    def __init__(self,Data):
        self.shape = (50,50)
        self.robots = 10
        self.grid = np.zeros(self.shape)-1
        self.epsilon = 0.1

        self.gener_info = self.createAllNodes()
        self.sci = 0.999
        self.Enviro_data = Data
        self.probabilities = self.Enviro_data.probabilities

        #sampling data from sensors
        self.sample_human()
        self.sample_elev()
        self.sample_temp()

        
    class Node:
        def __init__(self, x, y):
            self.x = x  # index of grid
            self.y = y  # index of grid

        def __repr__(self): 
            return "Node x:% s y:% s sam_temp:% s sam_elev:% s sam_human_pres:% s" % (self.x, self.y, self.sam_temp, self.sam_elev, self.sam_human_pres)

    def createAllNodes(self):
        Nodes = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                Nodes.append(self.Node(i,j))

        return Nodes
    
    def initialize(self):
        pos = []
        x_pos = random.sample(range(0, 49), self.robots)
        y_pos = random.sample(range(0, 49), self.robots)
        for i in range(self.robots):
            Sensor_placed = self.Node(x_pos[i],y_pos[i])
            pos.append(Sensor_placed) 
        
        return pos
    
    def sample_human(self):
        in_position = self.initialize()
        for i in range(self.robots):
            p = self.probabilities[in_position[i].x,in_position[i].y]*self.sci + (1-self.probabilities[in_position[i].x,in_position[i].y])*(1-self.sci)
            self.grid[in_position[i].x][in_position[i].y] = np.random.binomial(1, p, size=None)
        
        plt.imshow(self.grid,cmap='terrain')
        plt.show()
    
    def sample_elev(self):
        pass

    def sample_temp(self):
        pass

# if __name__ == '__main__':
#     Features = Sampling()  