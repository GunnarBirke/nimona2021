import numpy as np

class Gradient:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, t, y):
        return self.alpha * y

def euler(y0, f, timesteps):
    # assume that we got more than one step, otherwise there will be no deltat
    deltat = timesteps[1]
    y = np.zeros((timesteps.size, y0.shape[0]))
    y[0, :] = y0
    
    for i in range(0, timesteps.size - 1):
        t = deltat * i
        y[i+1, :] = y[i, :] + deltat * f(t, y[i, :])
        
    return y