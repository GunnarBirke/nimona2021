import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow
import numpy as np

class AnimateFunctor:
    def __init__(self, y, ax_circle, ax_map, positions, points, arrowPatch):
        self.arrowPatch = arrowPatch
        self.points = points
        self.y = y
        self.ax_circle = ax_circle
        self.ax_map = ax_map
        self.positions = positions
        
    def __call__(self, i):
        self.points.set_data(np.sin(self.y[i, :]), np.cos(self.y[i, :]))
    
        xCoord = np.sin(self.y[i, :]).sum()
        yCoord = np.cos(self.y[i, :]).sum()

        xCoord /= self.y[i, :].shape[0]
        yCoord /= self.y[i, :].shape[0]

        self.arrowPatch.remove()
        orderPoint = FancyArrow(0.0, 0.0, xCoord, yCoord, head_width = 0.05, color='orange', zorder=100.0)
        self.arrowPatch = self.ax_circle.add_patch(orderPoint)
        
        self.ax_map.clear()
        ret = self.ax_map.scatter(self.positions[:, 0], self.positions[:, 1], c=self.y[i, :]%(2*np.pi), cmap='hsv', vmin=0, vmax=2*np.pi)
    
        return self.points, self.arrowPatch, ret

def plotPhases(y, t, N, positions, title=""):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[4,1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_circle = fig.add_subplot(gs[0, 1])
    ax_synchro = fig.add_subplot(gs[1, 1])

    ax_circle.set_aspect('equal', 'box')
    circle = np.linspace(0, 2 * np.pi, 100)
    ax_circle.plot(np.sin(circle), np.cos(circle))

    points, = ax_circle.plot(np.sin(y[0, :]), np.cos(y[0, :]), '.r')
    
    xCoord = np.sin(y[0, :]).sum()
    yCoord = np.cos(y[0, :]).sum()
    
    xCoord /= N
    yCoord /= N
    
    orderPoint = FancyArrow(0.0, 0.0, xCoord, yCoord, head_width = 0.05, color='orange', zorder=100.0, length_includes_head=True)
    orderPointPatch = ax_circle.add_patch(orderPoint)
    
    animate = AnimateFunctor(y, ax_circle, ax_map, positions, points, orderPointPatch)

    def init():
        points.set_data(np.sin(y[0, :]), np.cos(y[0, :]))
        ret = ax_map.scatter(positions[:, 0], positions[:, 1], c=y[0, :]%(2*np.pi), cmap='hsv', vmin=0, vmax=2*np.pi)
        return points, orderPointPatch, ret

    _ = animation.FuncAnimation(plt.gcf(), animate, t.shape[0], init_func=init,
                                interval=500,
                                blit=True, repeat_delay=1000)

    plt.title('Test')
    plt.show()
    