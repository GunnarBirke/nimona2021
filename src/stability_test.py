import numpy as np
import numode
import matplotlib.pyplot as plt

class Gradient:
    def __call__(self, t, y):
        return np.array([-y[1], y[0]])
    
y0 = np.array([1.0, 0.0])
deltat = 0.01
T = 25 * np.pi
grad = Gradient()

t = np.arange(0, T + deltat, deltat)
y = numode.heun(y0, grad, t)
x = y[:, 0]
y = y[:, 1]

plt.plot(t, y)
#plt.plot(t, y)
plt.show()