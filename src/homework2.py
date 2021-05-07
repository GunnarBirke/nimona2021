import numpy as np
import numode as numode
import matplotlib.pyplot as plt

class Gradient:
    def __init__(self, epsilon1, epsilon2, gamma1, gamma2):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
    def __call__(self, t, y):
        return np.array([y[0] * (self.epsilon1 - self.gamma1 * y[1]), 
                         -y[1] * (self.epsilon2 - self.gamma2 * y[0])])



y0 = np.array([2000.0, 400.0])
deltat = 0.025
T = 150.0
grad = Gradient(2.0, 0.8, 0.02, 0.0002)

t = np.arange(0, T + deltat, deltat)
y = numode.heun(y0, grad, t)
yprey = y[:, 0]
ypredator = y[:, 1]

plt.plot(t, yprey)
plt.plot(t, ypredator)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.show()
    