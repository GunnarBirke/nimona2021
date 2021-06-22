import numpy as np
import json as json

config = None
simulationTopDirectory = './simulations'

with open(simulationTopDirectory + '/config.json', 'r') as f:
    config = json.load(f)
    
networkFilename = '/network.npy'

positions = np.load('./networks/ukPos.npy')
adj = np.load(simulationTopDirectory + networkFilename)

deltat = config['deltat']
T = config['T']
t = np.arange(0, T + deltat, deltat)

for i in range(0, 100):
    simulationDataFilename = '/simulationData' + str(i) + '.npy'

    y = np.load(simulationTopDirectory + simulationDataFilename)
    yn = y[y.shape[0] - 1, :]
    
    xOrderCoord = np.sin(yn).sum()
    yOrderCoord = np.cos(yn).sum()
    
    xOrderCoord /= yn.shape[0]
    yOrderCoord /= yn.shape[0]
    
    print(i, np.sqrt(xOrderCoord * xOrderCoord + yOrderCoord * yOrderCoord))
    
    