import numpy as np
import numode as numode
import network_model as network_model
import plotting as plot
import sys
import os
import json as json
    
if not os.path.isdir('./simulations'):
    os.mkdir('./simulations')
    
simulationTopDirectory = 'simulations'
    
config = {}

config['edgesFailingAtTime'] = []
config['edgesFailingAtSynchronization'] = []
config['deltat'] = 0.001
config['T'] = 1.5

positions = np.load('./networks/ukPos.npy')
adj = 50 * np.load('./networks/ukAdj.npy')

plot.plotMap(positions, adj, config)

with open('./' + simulationTopDirectory + '/config.json', 'w') as f:
    json.dump(config, f, indent=2)  
    
with open('./' + simulationTopDirectory + '/network.npy', 'wb') as f:
    np.save(f, adj)  

for i in range(0, 100):
    network = network_model.NetworkModel(config, adj, np.random.normal(loc=0.2, scale=0.01, size=adj.shape[0]))
    
    y0 = np.random.uniform(0.0, 2 * np.pi, network.adj.shape[0])      

    t = np.arange(0, config['T'] + config['deltat'], config['deltat'])
    y = numode.runge_kutta(y0, network, t)   

    with open(simulationTopDirectory+ '/simulationData' + str(i) + '.npy', 'wb') as f:
        np.save(f, y)