import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot
import sys
import json as json

config = None
simulationTopDirectory = './simulations'

with open(simulationTopDirectory + '/config.json', 'r') as f:
    config = json.load(f)

simulationDataFilename = '/simulationData0.npy'
networkFilename = '/network.npy'

positions = np.load('./networks/ukPos.npy')
adj = np.load(simulationTopDirectory + networkFilename)

deltat = config['deltat']
T = config['T']
t = np.arange(0, T + deltat, deltat)
y = np.load(simulationTopDirectory + simulationDataFilename)

start = t.shape[0]
start -= 100
start = 0

plot.plotPhases(y, t, adj.shape[0], positions, adj, config, start)