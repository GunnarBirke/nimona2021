import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot

network = network_model.NetworkModel()

positions = np.load('./networks/ukPos.npy')
network.adj = 50 * np.load('./networks/ukAdj.npy')

deltat = 0.001
T = 1.5
t = np.arange(0, T + deltat, deltat)
y = np.load('./simulations/uk_random.npy')

plot.plotPhases(y, t, network.adj.shape[0], positions)