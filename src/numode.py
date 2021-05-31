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

def runge_kutta(y0, f, timesteps):
    # assume that we got more than one step, otherwise there will be no deltat
    deltat = timesteps[1]
    y = np.zeros((timesteps.size, y0.shape[0]), order='C')
    y[0, :] = y0
    
    for i in range(0, timesteps.size - 1):
        t = deltat * i
        
        tangent = 0.0
        ycurr = y[i, :]
        
        eta1 = ycurr
        f1 = f(t, eta1)
        tangent += (1.0 / 6.0) * f1
        
        eta2 = ycurr + deltat * 0.5 * f1
        f2 = f(t + 0.5 * deltat, eta2)
        tangent += (1.0 / 3.0) * f2
        
        eta3 = ycurr + deltat * 0.5 * f2
        f3 = f(t + 0.5 * deltat, eta3)
        tangent += (1.0 / 3.0) * f3
        
        eta4 = ycurr + deltat * f3
        tangent += (1.0 / 6.0) * f(t + deltat, eta4)

        y[i+1, :] = ycurr + deltat * tangent
        
    return y