import noise
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


class generate_features():
    def __init__(self):
        #temperature feature variables
        self.shape = (50,50) 
        self.scale_temp = 100.0
        self.octaves_temp = 6
        self.persistence_temp = 0.75
        self.lacunarity_temp = 2.0
        self.node_set = self.createNodes()

        #Elevation feature variables
        self.shape = (50,50) 
        self.scale_elev = 100.0
        self.octaves_elev = 6
        self.persistence_elev = 0.75
        self.lacunarity_elev = 2.0

        #human probab features
        self.alpha = 0.01
        self.beta = 5
        self.tasks_human = 0
        self.probabilities = np.zeros(self.shape)

        #For loss of energy associated with the environment features
        self.frac_loss = self.loss_E()

        self.temp_dis = self.feature_temp()
        self.elev_dis = self.feature_elevation()
        self.human_dis = self.feature_human_probab()

    def loss_E(self):
        loss = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tem = random.uniform(0.001,0.2)
                elev = random.uniform(0.001,0.2)
                loss.append([tem,elev])
        return loss

    def createNodes(self):
        Nodes = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                Nodes.append(self.Node(i,j))

        return Nodes

    class Node:
        def __init__(self, x, y):
            self.x = x  # index of grid
            self.y = y  # index of grid 
        def __repr__(self): 
            return "Node x:% s y:% s temp:% s elev:% s human_pres:% s" % (self.x, self.y, self.temp, self.elev, self.human_pres)
        
    def feature_temp(self): 
        temp = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                temp[i][j] = noise.pnoise2(i/self.scale_temp, 
                                            j/self.scale_temp, 
                                            octaves=self.octaves_temp, 
                                            persistence=self.persistence_temp, 
                                            lacunarity=self.lacunarity_temp, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)
                self.node_set[i*self.shape[0] + j].temp = temp[i][j]
        
        plt.imshow(temp,cmap='inferno')
        # plt.show()
    
    def feature_elevation(self): 
        elev = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                elev[i][j] = noise.pnoise2(i/self.scale_elev, 
                                            j/self.scale_elev, 
                                            octaves=self.octaves_elev, 
                                            persistence=self.persistence_elev, 
                                            lacunarity=self.lacunarity_elev, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)
                self.node_set[i*self.shape[0] + j].elev = elev[i][j]
        
        plt.imshow(elev,cmap='terrain')
        # plt.show()

    def feature_human_probab(self):
        human_pres = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                p = np.random.beta(self.alpha,self.beta)
                self.probabilities[i][j]=p
                human_pres[i][j] = np.random.binomial(1, p, size=None)
                if human_pres[i][j]==1:
                    self.tasks_human+=1
                self.node_set[i*self.shape[0] + j].human_pres = human_pres[i][j]

        plt.imshow(human_pres,cmap='terrain')
        # plt.show()

# if __name__ == '__main__':
#     Features = generate_features()
    

    
