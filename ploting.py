import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import evol_strat_cma as es
import time

#Define the function you want to find the minimum of (here is a VERY simple one)
#If you know by advance a unique global minimum you can compare performances with other functions

def f(x):
    x,y = x[0], x[1]
    return 100*((y-x*x)**2) + (1-x)**2


fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

x = np.arange(-10, 10.1, 0.2)
y = np.arange(-10, 10.1, 0.2)

def f_to_z(xar, yar):
    ni = len(xar)
    nj = len(xar[0])
    zar = np.zeros((ni, nj))
    for i in range(0, ni):
        for j in range(0, nj):
            zar[i][j]=f((xar[i][j], yar[i][j]))
    return zar


X, Y = np.meshgrid(x, y)
Z = f_to_z(X,Y)

surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()     

start = time.time()

argmin = es.cma_es([[6],[-3]],1,0,f,10**(-10))

stop = time.time()

print("Found the minimum in ", stop-start, " s with a relative error of ", 100*np.linalg.norm(argmin - np.array([1.0,1.0]), 2)/np.linalg.norm(np.array([1.0,1.0]), 2), "%")

