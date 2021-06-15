import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot

network = network_model.NetworkModel()

positions = np.load('./networks/ukPos.npy')
network.adj = 50 * np.load('./networks/ukAdj.npy')

plot.plotMap(positions)

network.freqs = np.random.normal(loc=0.2, scale=0.01, size=network.adj.shape[0])
y0 = np.random.uniform(0.0, 2 * np.pi, network.adj.shape[0])

deltat = 0.001
T = 1.5
t = np.arange(0, T + deltat, deltat)
y = numode.runge_kutta(y0, network, t)

with open('./simulations/uk_random.npy', 'wb') as f:
    np.save(f, y)