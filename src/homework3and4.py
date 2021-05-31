import numpy as np
import numode as numode
import network_model as network_model

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time


network = network_model.NetworkModel()

network.adj = np.zeros((50, 50))

for i in range(network.adj.shape[0]):
    for j in range(i):
        network.adj[i, j] = 50.0
        network.adj[j, i] = 50.0
        
network.freqs = np.random.uniform(0.0, 2 * np.pi, 50)
y0 = np.random.uniform(0.0, 2 * np.pi, 50)

start = time.time()
print("hello")

deltat = 0.001
T = 150.0
t = np.arange(0, T + deltat, deltat)
y = numode.runge_kutta(y0, network, t)

end = time.time()
print("hello again")
print(end - start)

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
circle = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.sin(circle), np.cos(circle))

points, = ax.plot(np.sin(y0), np.cos(y0), '.r')

def animate(i):
    points.set_data(np.sin(y[i, :]), np.cos(y[i, :]))
    return points,

def init():
    points.set_data(np.sin(y[0, :]), np.cos(y[0, :]))
    return points,

_ = animation.FuncAnimation(plt.gcf(), animate, t.shape[0], init_func=init,
                            interval=200,
                            blit=True, repeat_delay=1000)

plt.title('Test')
plt.show()
    