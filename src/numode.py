import numpy as np


def euler(y0, f, timesteps):
    # assume that we got more than one step, otherwise there will be no deltat
    deltat = timesteps[1]
    y = np.zeros((timesteps.size, y0.shape[0]))
    y[0, :] = y0
    
    for i in range(0, timesteps.size - 1):
        t = deltat * i
        y[i+1, :] = y[i, :] + deltat * f(t, y[i, :])
        
    return y

def heun(y0, f, timesteps):
    # assume that we got more than one step, otherwise there will be no deltat
    deltat = timesteps[1]
    y = np.zeros((timesteps.size, y0.shape[0]))
    y[0, :] = y0
    
    for i in range(0, timesteps.size - 1):
        t = deltat * i
        z = y[i, :] + deltat * f(t, y[i, :])
        y[i+1, :] = y[i, :] + 0.5 * deltat * (f(t, y[i, :]) + f(t + deltat, z))
        
    return y