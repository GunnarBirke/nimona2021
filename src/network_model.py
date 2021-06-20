import numpy as np

class NetworkModel:
    def __init__(self, config):
        # pick some random initial data
        self.adj = np.zeros((10, 10))
        self.freqs = np.zeros((10))
        
        self.config = config
        self.nextFailingEdgeAtTime = 0
        self.nextFailingEdgeAtSynchronization = 0
        
    def __call__(self, t, y):
        grad = np.zeros(y.shape)
        
        while self.nextFailingEdgeAtTime < len(self.config['edgesFailingAtTime']) and self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['time'] <= t:
            self.adj[self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node1'], self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node2']] = 0.0
            self.adj[self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node2'], self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node1']] = 0.0
            self.nextFailingEdgeAtTime += 1
            
        while self.nextFailingEdgeAtSynchronization < len(self.config['edgesFailingAtSynchronization']) and self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['sync'] <= t:
            self.adj[self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node1'], self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node2']] = 0.0
            self.adj[self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node2'], self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node1']] = 0.0
            self.nextFailingEdgeAtSynchronization += 1
        
        for i in range(grad.size):
            grad[i] += self.freqs[i]
            
            for j in range(y.size):
                grad[i] += self.adj[i, j] * np.sin(y[j] - y[i])
                
        return grad