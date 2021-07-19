import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot
import sys
import os
import json as json

simulationTopDirectory = 'simulationsInertia0-1-2'
    
if not os.path.isdir('./' + simulationTopDirectory):
    os.mkdir('./' + simulationTopDirectory)
    
config = {}

config['edgesFailingAtTime'] = []
config['edgesFailingAtSynchronization'] = []
config['deltat'] = 0.01
config['T'] = 30.0
config['inertia'] = True
config['gamma'] = 0.1

positions = np.load('./networks/ukPos.npy')
adj = 50.0 * np.load('./networks/ukAdj.npy')

plot.plotMap(positions, adj, config)

with open('./' + simulationTopDirectory + '/config.json', 'w') as f:
    json.dump(config, f, indent=2)  
    
with open('./' + simulationTopDirectory + '/network.npy', 'wb') as f:
    np.save(f, adj)  

for i in range(0, 100):
    network = network_model.NetworkModel(config, adj, np.random.normal(loc=0.2, scale=0.01, size=adj.shape[0]))
    y0 = np.random.uniform(0.0, 2 * np.pi, network.adj.shape[0])
    
    if config['inertia']:
        freqs = 0.2 * (2 * np.random.binomial(n=1, p=0.5, size=adj.shape[0]) - 1)
        network = network_model.NetworkModelWithInertia(config, adj, freqs, config['gamma'])
        
        y0 = np.random.uniform(0.0, 2 * np.pi, 2 * network.adj.shape[0])
        
        for j in range(0, y0.size, 2):
            y0[j + 1] = 0

    t = np.arange(0, config['T'] + config['deltat'], config['deltat'])
    y = numode.runge_kutta(y0, network, t)

    with open(simulationTopDirectory+ '/frequencies' + str(i) + '.npy', 'wb') as f:
        np.save(f, network.freqs)

    with open(simulationTopDirectory+ '/simulationData' + str(i) + '.npy', 'wb') as f:
        np.save(f, y)