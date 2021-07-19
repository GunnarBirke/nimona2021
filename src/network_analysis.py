import numpy as np
import json as json
import matplotlib.pyplot as plt

config = None
simulationTopDirectory = './simulationsInertia0-1-2'

with open(simulationTopDirectory + '/config.json', 'r') as f:
    config = json.load(f)
    
networkFilename = '/network.npy'

positions = np.load('./networks/ukPos.npy')
adj = np.load(simulationTopDirectory + networkFilename)
originalAdj = np.load('./networks/ukAdj.npy')

deltat = config['deltat']
T = config['T']
inertia = config['inertia']
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

networkCount = 100

for i in range(0, networkCount):
    simulationDataFilename = '/simulationData' + str(i) + '.npy'

    y = np.load(simulationTopDirectory + simulationDataFilename)
    print(y.shape)
    
    if inertia:
        y1 = np.zeros((y.shape[0], y.shape[1] // 2))

        for j in range(y1.shape[0]):
            for k in range(y1.shape[1]):
                y1[j, k] = y[j, 2 * k]
                
        y = y1
    
    yn = y[y.shape[0] - 1, :]
    
    xOrderCoord = np.sin(yn).sum()
    yOrderCoord = np.cos(yn).sum()
    
    xOrderCoord /= yn.shape[0]
    yOrderCoord /= yn.shape[0]
    
    orderParamNormalized = np.array([xOrderCoord, yOrderCoord])
    op = orderParam(yn)
    
    orderParams.append(op)
    
    print(i, op)
    
    if op > 0.85:
        exNodes = findExceptionalNodes(yn, orderParamNormalized)
        
        if not exNodes:
            syncCount += 1
        print(exNodes)
    
print('Erfolgreiche Synchronisationen', syncCount, float(syncCount) / networkCount)

originalEdgeDistance = 0.0

for i in range(0, positions.shape[0]):
    for j in range(i + 1, positions.shape[0]):
        if originalAdj[i, j] != 0:
            originalEdgeDistance += np.linalg.norm(positions[i] - positions[j])
    
print('Ursprüngliche Länge an Leitungen', originalEdgeDistance)        
additionalEdgeDistance = 0.0

for i in range(0, positions.shape[0]):
    for j in range(i + 1, positions.shape[0]):
        if adj[i, j] != 0 and originalAdj[i, j] == 0:
            additionalEdgeDistance += np.linalg.norm(positions[i] - positions[j])
            
print('Zusätzliche Länge an Leitungen', additionalEdgeDistance)

fig, ax = plt.subplots()

bins = np.arange(0.0, 1.05, 0.05)
ax.hist(orderParams, bins)
ax.set_xlabel('Ordungsparameter')
ax.set_ylabel('Anzahl Simulationen')
ax.set_xticks(bins, minor=False)

plt.show()