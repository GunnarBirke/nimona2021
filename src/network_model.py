import numpy as np

class NetworkModel:
    def __init__(self):
        # pick some random initial data
        self.adj = np.zeros((10, 10))
        self.freqs = np.zeros((10))
        
    def __call__(self, t, y):
        grad = np.zeros(y.shape)
        
        for i in range(grad.size):
            grad[i] += self.freqs[i]
            
            for j in range(y.size):
                grad[i] += self.adj[i, j] * np.sin(y[j] - y[i])
                
        return grad