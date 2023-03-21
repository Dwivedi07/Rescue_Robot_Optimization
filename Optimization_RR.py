import numpy as np
import math
import matplotlib.pyplot as plt
import random
from pulp import *
from mpl_toolkits.mplot3d import axes3d

class Solver():
    def __init__(self,EData,RData):
        self.TN = RData.N_RR
        self.RN = len(RData.target_coordinates)

        self.robot_set = RData.Robot_set
        self.task_energy = RData.Tasks_energy
        self.robot_energy = RData.Initial_Energy
        self.decay_rate = RData.Decay_constant

        self.per_E = EData.frac_loss
        
    def solve(self):
        problem = LpProblem("Optimal Robot set", LpMinimize)

        #define decision variable
        tirj = LpVariable.dicts("taski_robotj", [(i, j) for i in range(self.TN) for j in range(self.RN)], cat=LpBinary)

        #Define Constraints

        for i in range(self.TN):
            problem += lpSum([tirj[(i, j)] for j in range(self.RN)]) == 1
        for i in range(self.RN):
            problem += lpSum([tirj[(i, j)] for j in range(self.TN)]) <= 1

        for i in range(self.TN):
            Ei = self.task_energy
            for j in range(self.RN):
                Eoj = self.robot_energy[j]




