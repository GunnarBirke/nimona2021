import numpy as np

class NetworkModel:
    def __init__(self, config, adj, freqs):
        self.adj = adj
        self.freqs = freqs
        
        self.config = config
        self.nextFailingEdgeAtTime = 0
        self.nextFailingEdgeAtSynchronization = 0
        
    def failEdges(self, t):
        while self.nextFailingEdgeAtTime < len(self.config['edgesFailingAtTime']) and self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['time'] <= t:
            self.adj[self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node1'], self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node2']] = 0.0
            self.adj[self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node2'], self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node1']] = 0.0
            self.nextFailingEdgeAtTime += 1
            
        while self.nextFailingEdgeAtSynchronization < len(self.config['edgesFailingAtSynchronization']) and self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['sync'] <= t:
            self.adj[self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node1'], self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node2']] = 0.0
            self.adj[self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node2'], self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node1']] = 0.0
            self.nextFailingEdgeAtSynchronization += 1
        
    def __call__(self, t, y):
        self.failEdges(t)
        
        grad = np.zeros(y.shape)
        
        for i in range(grad.size):
            grad[i] += self.freqs[i]
            
            for j in range(y.size):
                grad[i] += self.adj[i, j] * np.sin(y[j] - y[i])
                
        return grad
    
class NetworkModelWithInertia(NetworkModel):
    def __init__(self, config, adj, freqs, inertia):
        super().__init__(config, adj, freqs)
        self.inertia = inertia
        
    def __call__(self, t, y):
        self.failEdges(t)
        
        grad = np.zeros(y.shape)
        
        # first order model resulting from a second order model with twice the variables
        for i in range(grad.size // 2):
            grad[2 * i + 1] -= y[2 * i + 1] * self.inertia
            grad[2 * i + 1] += self.freqs[i]
            
            for j in range(y.size // 2):
                grad[2 * i + 1] += self.adj[i, j] * np.sin(y[2 * j] - y[2 * i])
                
            #grad[2 * i + 1] /= self.inertia
                
            grad[2 * i] = y[2 * i + 1]
                
        return grad