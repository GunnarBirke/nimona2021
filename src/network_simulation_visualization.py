import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot
import sys
import json as json

config = None
simulationTopDirectory = './simulationsGutesNetzwerk'

with open(simulationTopDirectory + '/config.json', 'r') as f:
    config = json.load(f)

simulationDataFilename = '/simulationData3.npy'
networkFilename = '/network.npy'

positions = np.load('./networks/ukPos.npy')
#positions = np.array([[0.5, 0.5], [0.7, 0.8], [0.6, 0.7]])
adj = np.load(simulationTopDirectory + networkFilename)

deltat = config['deltat']
T = config['T']
inertia = config['inertia']
t = np.arange(0, T + deltat, deltat)
y = np.load(simulationTopDirectory + simulationDataFilename)

if inertia:
    y1 = np.zeros((y.shape[0], y.shape[1] // 2))

    for i in range(y1.shape[0]):
        for j in range(y1.shape[1]):
            y1[i, j] = y[i, 2 * j]
            
    y = y1

start = t.shape[0]
start -= 100
#start = 0

plot.plotPhases(y, t, adj.shape[0], positions, adj, config, start)
