import numpy as np
import math
import matplotlib.pyplot as plt
import random
from pulp import *
from itertools import product
from mpl_toolkits.mplot3d import axes3d

class Solver():
    def __init__(self,EData,RData):
        self.TN = RData.N_RR
        self.RN_Nodes = RData.target_coordinates
        self.RN = len(RData.target_coordinates)
        self.path_TargetNode = RData.Path

        self.robot_set = RData.Robot_set
        self.task_energy = RData.Tasks_energy
        self.robot_energy = RData.Initial_Energy
        self.decay_rate = RData.Decay_constant

        self.per_E = np.array(EData.frac_loss)
        # print(self.per_E.shape)
        self.solve()
        
    def solve(self):
        problem = LpProblem("OptimalRobotset", LpMinimize)

        #define decision variable
        tirj = LpVariable.dicts("taski_robotj", [(i, j) for i in range(self.TN) for j in range(self.RN)], cat=LpBinary)

        #Define Constraints

        for i in range(self.TN):
            problem += lpSum([tirj[(i, j)] for j in range(self.RN)]) == 1
        for i in range(self.RN):
            problem += lpSum([tirj[(j, i)] for j in range(self.TN)]) <= 1

        coeffij = np.zeros((self.RN,self.TN))
        for i in range(self.RN):
            Ei = self.task_energy
            ni = len(self.path_TargetNode[i])
            for j in range(self.TN):
                lamj = self.decay_rate[j]
                Eoj = self.robot_energy[j]
                coeff_loss = self.calc_loss_trans(i)
                coeffij[i,j]= coeff_loss*Eoj + Ei + lamj*(ni-1)
                problem += tirj[(i, j)]*(Eoj-(coeff_loss*Eoj + Ei + lamj*(ni-1))) >= 0
                print(self.TN,i,j)

        #Define Objective Function
        problem += lpSum([tirj[(i, j)]*coeffij[i,j] for i in range(self.TN) for j in range(self.RN)])

        #Solve the problem
        problem.solve()

        #Print the solution
        for i in range(self.TN):
            row = ""
            for j in range(self.RN):
                if value(tirj[(i, j)]) == 1:
                    row += "Q "
                else:
                    row += ". "
            print(row)

    def calc_loss_trans(self,i):
        coeff = 1
        if len(self.path_TargetNode[i]) != 0:
            for j in range(len(self.path_TargetNode[i])):
                a=self.path_TargetNode[i][j][0]
                b=self.path_TargetNode[i][j][1]
                m=self.per_E[0][0]+self.per_E[0][1]
                SumLoss = m
                coeff = coeff*SumLoss
            return coeff
        else:
            return 0


