import random

a=[]
N =25-4
for i in range(4):
    if N!=0:
        gen = random.randint(0,N)
        N= N-gen
    else:
        gen = 0
    a.append(1+gen)


print(a)