import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot

network = network_model.NetworkModel()

positions = np.load('./networks/northernPos.npy')
network.adj = 50 * np.load('./networks/northernAdj.npy')

network.freqs = np.random.normal(loc=0.2, scale=0.01, size=50)
y0 = np.random.uniform(0.0, 2 * np.pi, 50)

deltat = 0.001
T = 1.5
t = np.arange(0, T + deltat, deltat)
y = numode.runge_kutta(y0, network, t)

plot.plotPhasesOnCircle(y, t, network.adj.shape[0])