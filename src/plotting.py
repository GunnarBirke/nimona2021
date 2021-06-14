import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
import numpy as np

class AnimateFunctor:
    def __init__(self, y, ax, points, arrowPatch):
        self.arrowPatch = arrowPatch
        self.points = points
        self.y = y
        self.ax = ax
        
    def __call__(self, i):
        self.points.set_data(np.sin(self.y[i, :]), np.cos(self.y[i, :]))
    
        xCoord = np.sin(self.y[i, :]).sum()
        yCoord = np.cos(self.y[i, :]).sum()

        xCoord /= self.y[i, :].shape[0]
        yCoord /= self.y[i, :].shape[0]

        self.arrowPatch.remove()
        orderPoint = FancyArrow(0.0, 0.0, xCoord, yCoord, head_width = 0.05, color='orange', zorder=100.0)
        self.arrowPatch = self.ax.add_patch(orderPoint)
    
        return self.points, self.arrowPatch

def plotPhasesOnCircle(y, t, N, title=""):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.sin(circle), np.cos(circle))

    points, = ax.plot(np.sin(y[0, :]), np.cos(y[0, :]), '.r')
    
    xCoord = np.sin(y[0, :]).sum()
    yCoord = np.cos(y[0, :]).sum()
    
    xCoord /= N
    yCoord /= N
    
    orderPoint = FancyArrow(0.0, 0.0, xCoord, yCoord, head_width = 0.05, color='orange', zorder=100.0, length_includes_head=True)
    orderPointPatch = ax.add_patch(orderPoint)
    
    animate = AnimateFunctor(y, ax, points, orderPointPatch)

    def init():
        points.set_data(np.sin(y[0, :]), np.cos(y[0, :]))
        return points, orderPointPatch

    _ = animation.FuncAnimation(plt.gcf(), animate, t.shape[0], init_func=init,
                                interval=500,
                                blit=True, repeat_delay=1000)

    plt.title('Test')
    plt.show()