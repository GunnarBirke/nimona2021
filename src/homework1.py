import numpy as np
import euler as euler
import matplotlib.pyplot as plt

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
    grad1 = euler.Gradient(-1.0)

    t = np.arange(0, T + deltat, deltat)
    y = euler.euler(y0, grad1, t)
    yscalars = np.concatenate(y)
    yexact = 0.5 * np.exp(-t)
    plotFunction(t, yscalars, yexact)
    
part1()