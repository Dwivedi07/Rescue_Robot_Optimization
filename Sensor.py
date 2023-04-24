#Main file
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

    def sample_pos(self):
        #Intitalizing four random parent population
        pop_i = []
        for i in range(self.num_parents):
            pop_i.append(self.initialize())
        

        CArea0 = [self.EA(pop_i[i]) for i in range(self.num_parents)]
        while (np.amax(CArea0)<0.9*(self.shape[0]*self.shape[1])):
            p1 = pop_i[np.argsort(CArea0)[-1]]
            p2 = pop_i[np.argsort(CArea0)[-2]]
            Cc = self.crossover(p1,p2)
            Mp = self.Mutate(p1,p2)

            pop_i = [Mp[0], Mp[1], Cc[0], Cc[1]]
            CArea0=[self.EA(pop_i[i]) for i in range(self.num_parents)]
            print('Maximum Area',CArea0[np.argsort(CArea0)[-1]])
        return pop_i[np.argsort(CArea0)[-1]]

    def crossover(self,p1,p2):
        n=len(p1)-1
        # print('size of list',n)
        a = int(np.random.uniform(1, n))
        # a = random.sample(range(1,n), 1)
        c1=[]
        c2=[]
        for i in range(len(p1)):
            if (i < n+1-a):
                c1.append(p1[i])
                c2.append(p2[i])
            else:
                c1.append(p2[i])
                c2.append(p1[i])
        
        return [c1, c2]

    def Mutate(self,p1,p2):
        numer_of_nodes_to_mutate = int(np.random.uniform(1, len(p1)))
        # print('numer_of_nodes_to_mutate',numer_of_nodes_to_mutate)

        indexes1 = random.sample(range(0, len(p1)-1), numer_of_nodes_to_mutate)
        # print(indexes1)
        for j in indexes1:
            p1[j].x = int(np.random.uniform(0, len(p1)-1))  #random.sample(range(0, len(p1)-1))
            p1[j].y = int(np.random.uniform(0, len(p1)-1)) 

        indexes2 = random.sample(range(0, len(p1)-1), numer_of_nodes_to_mutate)
        for j in indexes2:
            p2[j].x = int(np.random.uniform(0, len(p2)-1))
            p2[j].y = int(np.random.uniform(0, len(p2)-1))

        return [p1,p2]

    def EA(self,pos):
        area = self.shape[0]*self.shape[1]
        count = 0 
        for i in range(0, self.number_of_random_samples):
            x = np.random.uniform(0, self.shape[0])
            y = np.random.uniform(0, self.shape[1])
            temp_cover = self.is_point_under_coverage(pos,x,y)
            if (temp_cover): 
                count = count + 1
        estimated_area = area*count/self.number_of_random_samples
        # print("The estimated area is:", estimated_area)
        return estimated_area

    def is_point_under_coverage(self,pos,x,y):
        yes_it_is = False
        for i in range(0, self.srobots): 
            if np.sqrt((x - pos[i].x)**2 + (y - pos[i].y)**2) <= self.r[i]:
                yes_it_is = True
                break
        return yes_it_is

if __name__ == '__main__':
    Features = Sampling()  