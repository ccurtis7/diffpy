import numpy as np
import numpy.ma as ma
import pandas as pd


def walk1D(scale=1.0, steps=1000, drift=0.0):
    x = np.cumsum(np.random.normal(loc=drift, scale=scale, size=steps))
    return x


def walk2D(scale=(1.0, 1.0), steps=1000, drift=(0.0, 0.0), theta=0):
    theta = np.radians(theta)
    rMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.cumsum(np.random.normal(loc=drift[0], scale=scale[0], size=steps)).reshape((1, steps))
    y = np.cumsum(np.random.normal(loc=drift[1], scale=scale[1], size=steps)).reshape((1, steps))
    x = np.concatenate((x, y), axis=0)

    x = np.matmul(rMatrix, x)
    x, y = x[0, :], x[1, :]
    return x, y


def walks1D(scale=1.0, steps=1000, drift=0.0, N=10):
    x = np.cumsum(np.random.normal(loc=drift, scale=scale, size=(steps, N)), axis=0)
    return x


def walks2D(scale=(1.0, 1.0), steps=1000, startrange=(5,5), drift=(0.0, 0.0), theta=0, N=10, masked=False):
    if startrange:
        xlo, xhi, ylo, yhi = -startrange[0], startrange[0], -startrange[1], startrange[1]
        xstart, ystart = np.random.uniform(xlo,xhi,(1,N)), np.random.uniform(xlo,xhi,(1,N))
        xstart, ystart = np.tile(xstart, reps=(steps,1)), np.tile(ystart, reps=(steps,1))
    else:
        xstart, ystart = 0, 0
    
    theta = np.radians(theta)
    rMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.cumsum(np.random.normal(loc=drift[0], scale=scale[0], size=(steps, N)), axis=0).reshape((steps, 1, N))
    y = np.cumsum(np.random.normal(loc=drift[1], scale=scale[1], size=(steps, N)), axis=0).reshape((steps, 1, N))
    x = np.concatenate((x, y), axis=1)

    x = np.matmul(rMatrix, x)
    x, y = x[:, 0, :] + xstart, x[:, 1, :] + ystart

    if masked:
        xi, yi = np.meshgrid(np.arange(0, N), np.arange(0, steps))
        ind = np.random.randint(int(0.75*steps), steps, size=(1, N))
        ind = np.tile(ind, reps=(steps, 1))
        mask = ind < yi
        #x, y = ma.array(x, mask=mask), ma.array(y, mask=mask)
        x[mask], y[mask] = np.nan, np.nan

    return x, y


def walks2D(scale=(1.0, 1.0), steps=1000, drift=(0.0, 0.0), startrange=(5,5), circles=(0.1, 1), theta=0, N=10, masked=False):
    if startrange:
        xlo, xhi, ylo, yhi = -startrange[0], startrange[0], -startrange[1], startrange[1]
        xstart, ystart = np.random.uniform(xlo,xhi,(1,N)), np.random.uniform(xlo,xhi,(1,N))
        xstart, ystart = np.tile(xstart, reps=(steps,1)), np.tile(ystart, reps=(steps,1))
    else:
        xstart, ystart = 0, 0
    
    if circles:
        rperstep, r = circles[0], circles[1]
        rads = np.arange(0, rperstep*steps, rperstep)
        rstart = np.random.uniform(low=0.0, high=2*np.pi, size=N)
        radsx, rstarty = np.meshgrid(rstart, rads)
        rads = (radsx + rstarty).reshape(steps, 1, N)

        xc, yc = r*np.cos(rads), r*np.sin(rads)
    else:
        xc, yc = np.zeros((steps,1,N)), np.zeros((steps,1,N))
    
    theta = np.radians(theta)
    rMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.cumsum(np.random.normal(loc=drift[0], scale=scale[0], size=(steps, N)), axis=0).reshape((steps, 1, N)) + xc
    y = np.cumsum(np.random.normal(loc=drift[1], scale=scale[1], size=(steps, N)), axis=0).reshape((steps, 1, N)) + yc
    x = np.concatenate((x, y), axis=1)
    
    x = np.matmul(rMatrix, x)
    x, y = x[:, 0, :] + xstart, x[:, 1, :] + ystart

    if masked:
        xi, yi = np.meshgrid(np.arange(0, N), np.arange(0, steps))
        ind = np.random.randint(int(0.75*steps), steps, size=(1, N))
        ind = np.tile(ind, reps=(steps, 1))
        mask = ind < yi
        #x, y = ma.array(x, mask=mask), ma.array(y, mask=mask)
        x[mask], y[mask] = np.nan, np.nan

    return x, y


def trajStack(x, y):
    steps, N = x.shape
    xi, yi = np.meshgrid(np.arange(0, N), np.arange(0, steps))

    tStack = pd.DataFrame({'Number': xi.flatten(order='F'),
                           'Frame': yi.flatten(order='F'),
                           'x': x.flatten(order='F'),
                           'y': y.flatten(order='F')})
    tStack = tStack.set_index(['Number', 'Frame'])

    return tStack


def trajUnstack(data):
    x = data.unstack(level=0)['x'].to_numpy()
    y = data.unstack(level=0)['y'].to_numpy()

    return x, y
