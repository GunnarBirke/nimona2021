import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow
import numpy as np

class AnimateFunctor:
    def __init__(self, y, ax_circle, ax_map, positions, points, adj, arrowPatch, config, start):
        self.arrowPatch = arrowPatch
        self.points = points
        self.y = y
        self.ax_circle = ax_circle
        self.ax_map = ax_map
        self.positions = positions
        self.adj = adj
        self.t = 0
        self.config = config
        self.nextFailingEdgeAtTime = 0
        self.nextFailingEdgeAtSynchronization = 0
        self.start = start
        
    def __call__(self, i):
        i += self.start
        
        while self.nextFailingEdgeAtTime < len(self.config['edgesFailingAtTime']) and self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['time'] <= self.t:
            self.adj[self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node1'], self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node2']] = 0.0
            self.adj[self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node2'], self.config['edgesFailingAtTime'][self.nextFailingEdgeAtTime]['node1']] = 0.0
            self.nextFailingEdgeAtTime += 1
            
        while self.nextFailingEdgeAtSynchronization < len(self.config['edgesFailingAtSynchronization']) and self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['sync'] <= self.t:
            self.adj[self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node1'], self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node2']] = 0.0
            self.adj[self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node2'], self.config['edgesFailingAtSynchronization'][self.nextFailingEdgeAtSynchronization]['node1']] = 0.0
            self.nextFailingEdgeAtSynchronization += 1
            
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
        
        toUpdate = [self.points, self.arrowPatch, ret]
        
        for i in range(self.adj.shape[0]):
            for j in range(self.adj.shape[1]):
                if self.adj[i, j] != 0.0:
                    toUpdate.append(self.ax_map.plot(self.positions[[i,j], 0], self.positions[[i,j], 1], 'k-', zorder=0)[0])
    
        self.t += self.config['deltat']
        
        return toUpdate

def plotPhases(y, t, N, positions, adj, config, start, title=""):
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
    
    animate = AnimateFunctor(y, ax_circle, ax_map, positions, points, adj, orderPointPatch, config, start)

    def init():
        points.set_data(np.sin(y[0, :]), np.cos(y[0, :]))
        ret = ax_map.scatter(positions[:, 0], positions[:, 1], c=y[0, :]%(2*np.pi), cmap='hsv', vmin=0, vmax=2*np.pi)
        
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ax_map.plot(positions[[i,j], 0], positions[[i,j], 1], 'k-', zorder=0)
                
        return points, orderPointPatch, ret

    _ = animation.FuncAnimation(plt.gcf(), animate, t.shape[0] - start, init_func=init,
                                interval=10,
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
    
def drawMap(positions, adj, ax, config):
    plt.scatter(positions[:, 0], positions[:, 1], c='r', s=0.3)
    
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[1]):
            color = 'b-'
            
            for k in range(0, len(config['edgesFailingAtTime'])):
                if config['edgesFailingAtTime'][k]['node1'] == i and config['edgesFailingAtTime'][k]['node2'] == j:
                    color = 'k-'
                    
            for k in range(0, len(config['edgesFailingAtSynchronization'])):
                if config['edgesFailingAtSynchronization'][k]['node1'] == i and config['edgesFailingAtSynchronization'][k]['node2'] == j:
                    color = 'g-'
                    
            if adj[i, j] != 0:
                plt.plot(positions[[i,j], 0], positions[[i,j], 1], color, zorder=0)

firstPoint = -1
edgeAction = 0
nextEdgeFailingAtTime = 0
nextEdgeFailingAtSynchronization = 0

def plotMap(positions, adj, config):
    fig, ax = plt.subplots()
    c = np.zeros(positions.shape[0]) + 2
    ax.set_aspect('equal', 'box')
    
    def onclick(event):
        global firstPoint
        global edgeAction
        global nextEdgeFailingAtTime
        global nextEdgeFailingAtSynchronization
        
        x = event.xdata
        y = event.ydata
        
        i = findClosestPoint(positions, x, y)
        
        if firstPoint == -1:
            firstPoint = i
        else:
            if edgeAction == 0:
                if nextEdgeFailingAtTime < len(config['edgesFailingAtTime']):
                    config['edgesFailingAtTime'][nextEdgeFailingAtTime]['node1'] = firstPoint
                    config['edgesFailingAtTime'][nextEdgeFailingAtTime]['node2'] = i
                    nextEdgeFailingAtTime += 1
            elif edgeAction == 1:
                if nextEdgeFailingAtSynchronization < len(config['edgesFailingAtSynchronization']):
                    config['edgesFailingAtSynchronization'][nextEdgeFailingAtSynchronization]['node1'] = firstPoint
                    config['edgesFailingAtSynchronization'][nextEdgeFailingAtSynchronization]['node2'] = i
                    nextEdgeFailingAtSynchronization += 1
            elif edgeAction == 2:
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
        drawMap(positions, adj, ax, config)
        plt.draw()
        
    def onkeypress(event):
        global edgeAction
        
        if event.key == 't':
            edgeAction = 0
            print('Select edge to fail at a given time')
        elif event.key == 'y':
            edgeAction = 1
            print('Select edge to fail at a given synchronization state')
        elif event.key == 'c':
            edgeAction = 2
            print('Add or remove edges')
            
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkeypress)
    
    drawMap(positions, adj, ax, config)
    plt.show()