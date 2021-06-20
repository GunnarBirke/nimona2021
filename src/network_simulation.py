import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot
import sys
import os
import json as json

if not os.path.isdir('./networks'):
    os.mkdir('./networks')
    
if not os.path.isdir('./simulations'):
    os.mkdir('./simulations')
    
config = {}

config['simulationDataFilename'] = 'ukSimulation1.npy'
config['networkFilename'] = 'ukChangedNetwork1.npy'
config['edgesFailingAtTime'] = []
config['edgesFailingAtSynchronization'] = [{'sync': 0.9, 'node1': 0, 'node2': 0}]
config['deltat'] = 0.001
config['T'] = 1.5

network = network_model.NetworkModel(config)

positions = np.load('./networks/ukPos.npy')
network.adj = 50 * np.load('./networks/ukAdj.npy')

plot.plotMap(positions, network.adj, config)

network.freqs = np.random.normal(loc=0.2, scale=0.01, size=network.adj.shape[0])
y0 = np.random.uniform(0.0, 2 * np.pi, network.adj.shape[0])      

with open("simulations/config1.json", 'w') as f:
    json.dump(config, f, indent=2)  
    
with open('./networks/' + config['networkFilename'], 'wb') as f:
    np.save(f, network.adj)  

t = np.arange(0, config['T'] + config['deltat'], config['deltat'])
y = numode.runge_kutta(y0, network, t)   

with open('./simulations/' + config['simulationDataFilename'], 'wb') as f:
    np.save(f, y)