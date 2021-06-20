import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot
import sys
import json as json

config = None

with open("simulations/config1.json", 'r') as f:
    config = json.load(f)

simulationDataFilename = config['simulationDataFilename']
networkFilename = config['networkFilename']

network = network_model.NetworkModel(config)

positions = np.load('./networks/ukPos.npy')
network.adj = np.load('./networks/' + networkFilename)

deltat = config['deltat']
T = config['T']
t = np.arange(0, T + deltat, deltat)
y = np.load('./simulations/' + simulationDataFilename)

plot.plotPhases(y, t, network.adj.shape[0], positions, network.adj, config)