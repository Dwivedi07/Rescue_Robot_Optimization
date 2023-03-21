import random
from itertools import product
import numpy as np
a =[[1,2,3],[4,5,6],[3,3,3]]
p_t = np.array(a)
num_products = 3

rslt1 = sum (p_t[i, j] for i,j in product(range(num_products), repeat=2))

# a=[]
# N =25-4
# for i in range(4):
#     if N!=0:
#         gen = random.randint(0,N)
#         N= N-gen
#     else:
#         gen = 0
#     a.append(1+gen)
# a =  [sum(i+j) for i in range(10) for j in range(10)]

print(rslt1)
