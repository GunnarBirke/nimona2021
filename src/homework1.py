import numpy as np
import numode as numode
import matplotlib.pyplot as plt

class Gradient:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, t, y):
        return self.alpha * y
    
class Gradient2:
    def __init__(self, omega):
        self.omega = omega
        
    def __call__(self, t, y):
        return np.array([y[1], -self.omega**2 * y[0]])

def plotFunction(t, ynum, yexact):
    plt.plot(t, ynum)
    plt.plot(t, yexact)
    plt.xlabel('$t$')
    plt.ylabel('$y$')
    plt.show()

def part1():
    y0 = np.array([0.5])
    deltat = 0.5
    T = 7.0
    grad1 = Gradient(-1.0)

    t = np.arange(0, T + deltat, deltat)
    y = numode.euler(y0, grad1, t)
    yscalars = y[:, 0]
    yexact = 0.5 * np.exp(-t)
    plotFunction(t, yscalars, yexact)
    
def part2():
    y0 = np.array([0.0, 1.0])
    deltat = 0.05
    T = 20.0 * np.pi
    grad2 = Gradient2(-1.0)

    t = np.arange(0, T + deltat, deltat)
    y = numode.euler(y0, grad2, t)
    yscalars = y[:, 0]
    yexact = np.sin(t)
    plotFunction(t, yscalars, yexact)
    
part1()
part2()