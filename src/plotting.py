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
    
def distanceLineSegmentPoint(a, b, p):
    ab = b - a
    ap = p - a
    bp = p - b
    
    e = np.dot(ap, ab)
    
    if e <= 0.0:
        return np.dot(ap, ap)
    
    f = np.dot(ab, ab)
    
    if e >= f:
        return np.dot(bp, bp)
    
    return np.dot(ap, ap) - e * e / f
    
def findClosestEdge(positions, adj, x, y):
    minDist = 100000.0 # just pick a very large value and hope for the best
    minI = -1
    minJ = -1
    
    for i in range(0, positions.shape[0]):
        for j in range(0, positions.shape[0]):
            p1 = positions[i, :]
            p2 = positions[j, :]
            p = np.array([x, y])
            d = distanceLineSegmentPoint(p1, p2, p)
            
            if d < minDist:
                minDist = d
                minI = i
                minJ = j
                
    return (minI, minJ)

def findClosestPoint(positions, x, y):
    p = np.array([x, y])
    minDist = 100000.0 # again, we hope..
    pIndex = -1
    
    for i in range(0, positions.shape[0]):
        d = np.dot(positions[i] - p, positions[i] - p)
        
        if d < minDist:
            minDist = d
            pIndex = i
            
    return pIndex
    
def drawMap(positions, adj, ax):
    plt.scatter(positions[:, 0], positions[:, 1], c='r', s=0.3)
    
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                plt.plot(positions[[i,j], 0], positions[[i,j], 1], 'k-', zorder=0)

firstPoint = -1

def plotMap(positions, adj):
    fig, ax = plt.subplots()
    c = np.zeros(positions.shape[0]) + 2
    ax.set_aspect('equal', 'box')
    
    def onclick(event):
        global firstPoint
        x = event.xdata
        y = event.ydata
        
        i = findClosestPoint(positions, x, y)
        
        if firstPoint == -1:
            firstPoint = i
        else:
            if adj[firstPoint, i] == 0.0:
                adj[firstPoint, i] = 50.0
                adj[i, firstPoint] = adj[firstPoint, i]
            else:
                adj[firstPoint, i] = 0.0
                adj[i, firstPoint] = adj[firstPoint, i]
                
            firstPoint = -1
                
        
        plt.cla()
        event.inaxes.set_aspect('equal', 'box')
        plt.plot(x, y, 'k+')
        drawMap(positions, adj, ax)
        plt.draw()
        
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    drawMap(positions, adj, ax)
    plt.show()