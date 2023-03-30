import argparse, time
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
from Environment_generate import generate_features
from Sensor import Sampling
from Optimization_RR import Solver

class ProblemFeatures():
    def __init__(self,args,Data):
        self.N_RR = args.rescuerobots
        self.T_RR = args.typerr
        self.N_SR = args.sensorrobots
        self.T_SR = args.typesr
        self.Task_types = 1            #default rescue humans trapped


        self.node= Data.node_set       #Data generated from the environment sample planning
        self.Data = Data

        self.target_coordinates = self.task_nodes()
        self.Path = self.path_to_task_nodes()

        self.Robot_set = self.type_set()
        self.Initial_Energy = self.Initialize_energy()
        self.Decay_constant = self.Initialize_decayrate()
        self.Tasks_energy = 150
        
        

    def task_nodes(self):
        indexes = []
        '''
        Simplifying the model:Humans rescue is the only task to execute
        '''
        for i in range(len(self.node)):
            if self.node[i].human_pres == 1:     
                x = i//self.Data.shape[0]
                y = i%self.Data.shape[0]
                indexes.append([x,y])
        return indexes

    def type_set(self):
        a=[]
        N = self.N_RR-self.T_RR
        for i in range(self.T_RR):
            if i==self.T_RR-1:
                gen = N
                a.append(gen+1)
            elif N!=0:
                gen = random.randint(0,N)
                a.append(1+gen)
                N= N-gen
            else:
                gen = 0
                a.append(1+gen)
        
        return a

    def Initialize_energy(self):
        En=[]
        for i in range(self.T_RR):
            energy = random.uniform(175,250)
            for j in range(self.Robot_set[i]):
                En.append(energy)

        '''Testing::
        I have made first n(equal to number of tasks to do) robots to have max energy so ideally our answer should be those only
        '''
        # for i in range(len(self.target_coordinates)):
        #     En[i] = 250
        return En
                
    def Initialize_decayrate(self):
        lam=[]
        for i in range(self.T_RR):
            rate = random.uniform(0.6,2.6)
            for j in range(self.Robot_set[i]):
                lam.append(rate) 

        '''Testing::
        I have made first n(equal to number of tasks to do) robots to have 0 decay rate so ideally our answer should be those only
        '''
        # for i in range(len(self.target_coordinates)):
        #     lam[i] = 0
        return lam

    def get_path_points(self,xg,yg):
        pp=[]
        if xg!=0:
            m = yg/xg
            for i in range(xg+1):
                pp.append([i,int(m*i)])
        else:                                     #vertical path
            for i in range(yg+1):
                pp.append([0,i])

        return pp
    
    def path_to_task_nodes(self):
        path = []
        for i in range(len(self.target_coordinates)):
            path.append(self.get_path_points(self.target_coordinates[i][0],self.target_coordinates[i][1]))
        return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calamity', type=str, required=True, help='The calamity over which optimization run. Valid values are: Fire, Flood')
    # parser.add_argument('--tasks', type=int, required=True, help='Give number of tasks to perform.i.e..Number of umans trapped . Valid values are: integers!=0')
    parser.add_argument('--rescuerobots', type=int, required=True, help='Number of rescue robots. Valid values are: integers!=0')
    parser.add_argument('--typerr', type=int, required=True, help='Type  of rescue robots. Valid values are: integers!=0')
    parser.add_argument('--sensorrobots', type=int, required=True, help='Number of sensor robots. Valid values are: integers!=0')
    parser.add_argument('--typesr', type=int, required=True, help='Type of sensor robots. Valid values are: integers!=0')
    args = parser.parse_args()
    

    start = time.time()
    if args.calamity == 'Fire':
        print("="*18 + 'Task Fire Rescue' + "="*18)
        EnvironmentData = generate_features()
        RobotData = ProblemFeatures(args,EnvironmentData)
        # EnvironmentPlanner = Sampling(EnvironmentData)

        print('Number of Robots =',args.rescuerobots)
        print('Number of Tasks  =',len(RobotData.target_coordinates))
        Solution = Solver(EnvironmentData,RobotData)
        
        
    if args.calamity == 'Flood':
        print(f'=*{18}' + f'Task Flood Rescue' + f'=*{18}')
        
    
    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end-start))
