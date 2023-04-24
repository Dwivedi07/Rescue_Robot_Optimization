import numpy as np
import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
from Environment_generate import generate_features

class Sampling():
    def __init__(self):
        self.shape = (100,100)
        self.srobots = 20
        self.typerr = 2
        self.dis=[10,10]
        self.radius = [5,10]
        self.gener_info = self.createAllNodes()
        self.num_parents = 4
        self.number_of_random_samples = 10000
        self.r = self.in_r()
        self.sample_pos()
        
        
    def in_r(self):
        r = np.zeros(self.srobots)
        for i in range(self.dis[0]):
            r[i]= self.radius[0]
        for j in range(self.dis[0], self.dis[0]+self.dis[1]):
            r[j]= self.radius[1]
        
        return r

        
    class Node:
        def __init__(self, x, y):
            self.x = x  # index of grid
            self.y = y  # index of grid

        def __repr__(self): 
            return "Node x:% s y:% s" % (self.x, self.y)
            # return "Node x:% s y:% s sam_temp:% s sam_elev:% s sam_human_pres:% s" % (self.x, self.y, self.sam_temp, self.sam_elev, self.sam_human_pres)

    def createAllNodes(self):
        Nodes = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                Nodes.append(self.Node(i,j))

        return Nodes
    
    def initialize(self):
        pos = []
        x_pos = random.sample(range(0, self.shape[0]-1), self.srobots)
        y_pos = random.sample(range(0, self.shape[0]-1), self.srobots)
        for i in range(self.srobots):

            Sensor_placed = self.Node(x_pos[i],y_pos[i])
            pos.append(Sensor_placed) 
        return pos


if __name__ == '__main__':
    Features = Sampling()  