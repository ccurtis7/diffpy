import numpy as np

def trajDistance(x, y):
    return np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


def trajsDistance(x, y):
    return np.cumsum(np.sqrt(np.diff(x, axis=0)**2 + np.diff(y, axis=0)**2), axis=0)


def trajsMSD(x, y):
    steps, N = x.shape
    mx1, mx2 = np.zeros((steps, steps, N)), np.zeros((steps, steps, N))
    my1, my2 = np.zeros((steps, steps, N)), np.zeros((steps, steps, N))

    for i in range(N):
        mx1[:, :, i], mx2[:, :, i] = np.meshgrid(x[:, i], x[:, i])
        my1[:, :, i], my2[:, :, i] = np.meshgrid(y[:, i], y[:, i])
        
        
    mmsd = (mx1 - mx2)**2 + (my1 - my2)**2
    mmsddiag = np.ones((steps-1, steps-1, N))*np.nan
    
    mgaus = (mx1 - mx2)**4 + (my1 - my2)**4
    mgausdiag = np.ones((steps-1, steps-1, N))*np.nan

    for i in range(1, steps):
        mmsddiag[i-1, i-1:, :] = np.transpose(np.diagonal(mmsd, offset=i))
        mgausdiag[i-1, i-1:, :] = np.transpose(np.diagonal(mgaus, offset=i))
        
    msds = np.nanmean(mmsddiag, axis=1)
    gauss = 2*np.nanmean(mgausdiag, axis=1)/(3*msds**2) - 1
    
    return msds, gauss


def trajMSD(x, y):
    steps = x.shape[0]
    x, y = x.reshape((steps, 1)), y.reshape((steps, 1))
    msds, gauss = trajsMSD(x, y)
    return msds.reshape(steps-1), gauss.reshape(steps-1)

