import numpy as np
import numode
import matplotlib.pyplot as plt

class Gradient:
    def __init__(self):
        self.alpha = 1.0
        
    def __call__(self, t, y):
        return self.alpha * y

def plotArrowsHeun(t, y, f, deltat):
    z = y + deltat * f(t, y)
    plt.arrow(t, y, 0.5 * deltat, 0.5 * deltat * f(t, y), head_width = 0.1)
    plt.arrow(t + deltat, z, 0.5 * deltat, 0.5 * deltat * f(t + deltat, z), head_width = 0.1)
    plt.arrow(t, y, deltat, 0.5 * deltat * (f(t + deltat, z) + f(t, y)), head_width = 0.1)
    
    
def plotArrowsKutta(t, y, f, deltat):
    tangent = 0.0
        
    eta1 = y
    tangent1 = (1.0 / 6.0) * f(t, eta1)
    tangent += tangent1
    plt.arrow(t, eta1, (1.0 / 6.0) * deltat, deltat * tangent1, head_width = 0.05, color='orange', zorder=100.0)
    
    eta2 = y + deltat * 0.5 * f(t, eta1)
    tangent2 = (1.0 / 3.0) * f(t + 0.5 * deltat, eta2)
    tangent += tangent2
    plt.arrow(t + 0.5 * deltat, eta2, (1.0 / 3.0) * deltat, deltat * tangent2, head_width = 0.05, color='orange', zorder=100.0)
    
    eta3 = y + deltat * 0.5 * f(t + 0.5 * deltat, eta2)
    tangent3 = (1.0 / 3.0) * f(t + 0.5 * deltat, eta3)
    tangent += tangent3
    plt.arrow(t + 0.5 * deltat, eta3, (1.0 / 3.0) * deltat, deltat * tangent3, head_width = 0.05, color='orange', zorder=100.0)
    
    eta4 = y + deltat * f(t + 0.5 * deltat, eta3)
    tangent4 = (1.0 / 6.0) * f(t + deltat, eta4)
    tangent += tangent4
    plt.arrow(t + deltat, eta4, (1.0 / 6.0) * deltat, deltat * tangent4, head_width = 0.05, color='orange', zorder=100.0, label='RK4 Stufe')
    
    plt.arrow(t, y, deltat, deltat * tangent, head_width = 0.05, color='green', zorder=10.0, linestyle=':', label='RK4 Schritt')
    
y0 = np.array([1.0])
deltat = 0.5
T = 1.0
grad = Gradient()

t = np.arange(0, T + deltat, deltat)
t2 = np.arange(0, T, deltat * 0.01)
y = numode.runge_kutta(y0, grad, t)[:, 0]

plt.plot(t, y, label='RK4', zorder=0.1)
plt.plot(t2, np.exp(t2), label='exakt', color='red', zorder=0.0)

plotArrowsKutta(t[0], y[0], grad, t[1])

plt.legend()
plt.show()