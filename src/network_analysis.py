import numpy as np
import json as json
import matplotlib.pyplot as plt

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

def orderParam(ys):
    xOrderCoord = np.sin(ys).sum()
    yOrderCoord = np.cos(ys).sum()
    
    xOrderCoord /= ys.shape[0]
    yOrderCoord /= ys.shape[0]
    
    return np.sqrt(xOrderCoord * xOrderCoord + yOrderCoord * yOrderCoord)

def findExceptionalNodes(yn, orderParamNormalized):
    exceptionalNodes = []
    
    xCoord = np.sin(yn)
    yCoord = np.cos(yn)
    
    for i in range(0, yn.shape[0]):
        v = np.array([xCoord[i], yCoord[i]])
        
        if np.linalg.norm(v - orderParamNormalized) > 1.0:
            exceptionalNodes.append(i)
            
    return exceptionalNodes

def logisticCurve(y, deltat, op):
    i = int(y.shape[0] * (1.0 / 3.0))
    t = deltat * i
    ys = y[i, :]
    
    op2 = orderParam(ys)
    
    if op > op2:
        k = np.log(op / op2 - 1) / (-t)
    
        return k
    else:
        return 0.0
    
orderParams = []
syncCount = 0

for i in range(0,81):
    simulationDataFilename = '/simulationData' + str(i) + '.npy'

    y = np.load(simulationTopDirectory + simulationDataFilename)
    yn = y[y.shape[0] - 1, :]
    
    xOrderCoord = np.sin(yn).sum()
    yOrderCoord = np.cos(yn).sum()
    
    xOrderCoord /= yn.shape[0]
    yOrderCoord /= yn.shape[0]
    
    orderParamNormalized = np.array([xOrderCoord, yOrderCoord])
    op = orderParam(yn)
    
    orderParams.append(op)
    
    print(i, op, logisticCurve(y, deltat, op))
    
    if op > 0.85:
        exNodes = findExceptionalNodes(yn, orderParamNormalized)
        
        if not exNodes:
            syncCount += 1
        print(exNodes)
    
print(syncCount)
#fig, ax = plt.subplots()
#ax.hist(orderParams, bins=20)
#plt.show()