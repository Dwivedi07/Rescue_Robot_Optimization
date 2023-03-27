import numpy as np
import matplotlib.pyplot as plt
from math import *
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Sum_of_Gaussians: 
    def __init__(self, shape, n, mean, covariance):
        self.shape = shape
        self.gaussian_sum = np.zeros(self.shape)
        self.n = n 
        self.mean = mean
        self.covariance = covariance
        self.generate_sum = self.generate_sum()

    def generate_sum(self): 
        for i in range(0,self.n): 
            for j in range(0,self.shape[0]):
                for k in range(0, self.shape[1]): 
                    self.gaussian_sum[j][k] = self.gaussian_sum[j][k] + np.exp(-0.5*((j - self.mean[i][0])**2 + (k - self.mean[i][1])**2)/(self.covariance[i]**2))
        lin_x = np.linspace(0,1,self.shape[0],endpoint=False)
        lin_y = np.linspace(0,1,self.shape[1],endpoint=False)
        x,y = np.meshgrid(lin_x,lin_y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x,y,self.gaussian_sum,cmap='terrain')
        #t.imshow(self.gaussian_sum, cmap = 'terrain')
        plt.show()

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
    generate_gaussian_sum = Sum_of_Gaussians(a, n, mean, covariance)

    