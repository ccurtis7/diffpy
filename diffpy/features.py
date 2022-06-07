import sys
import os
sys.path.insert(0, os.path.abspath('..\\diffpy'))
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage.interpolation import rotate
from scipy.spatial import ConvexHull
import msds as msds

def anomalous(x, D, alpha, dt):
    # The anomalous diffusion model
    return 4*D*(x*dt)**alpha


def anomModels(MSD, dt, skips=1):
    # Calculates the anomalous diffusion model parameters for a single particle
    # from its MSD profile over multiple timespans
    # By default, fits the model to all possible MSD profiles of at least 10 frames

    def anom(x, D, alpha):
        return anomalous(x, D, alpha, dt=dt)

    steps = MSD.shape[0] + 1
    N = np.linspace(1, steps-1, steps-1)

    alphas = np.zeros(steps-1)
    Ds = np.zeros(steps-1)
    for i in range(10, steps-1, skips):
        ps, _ = curve_fit(anom, N[:i], MSD[:i], p0=[6,1])
        Ds[i] = ps[0]
        alphas[i] = ps[1]

    return Ds, alphas


def anomModel(MSD, dt, frame=100):
    # Calculates the anomalous diffusion model parameters for multiple
    # particle trajectories. The user specifies up to which frame to include
    # in the MSD trajectory to use in the model fit.

    def anom(x, D, alpha):
        return anomalous(x, D, alpha, dt=dt)

    steps = MSD.shape[0] + 1
    N = np.linspace(1, steps-1, steps-1)

    ps, _ = curve_fit(anom, N[:frame], MSD[:frame], p0=[6,1])
    D = ps[0]
    alpha = ps[1]

    return D, alpha


def anomModelN(MSD, dt, frame=100):
    # Calculate the anomalous diffusion model parameters for multiple particle
    # trajectories.

    def anom(x, D, alpha):
        return anomalous(x, D, alpha, dt=dt)

    steps, N = MSD.shape
    n = np.linspace(1, steps, steps)

    # For arrays that contain NaNs
    mask = MSD > -10000
    cutoff = steps*np.average(mask, axis=0)

    Ds = np.zeros(N)
    alphas = np.zeros(N)
    for i in range(N):
        cut = int(cutoff[i])
        if cut < frame:
            frame = cut
        ps, _ = curve_fit(anom, n[:frame], MSD[:frame, i], p0=[6,1])
        Ds[i] = ps[0]
        alphas[i] = ps[1]

    return Ds, alphas


def anomModelsN(MSD, dt, skips=1):
    # Calculate the anomalous diffusion model parameters for multiple particle
    # trajectories over multiple MSD window frames.

    def anom(x, D, alpha):
        return anomalous(x, D, alpha, dt=dt)

    steps, N = MSD.shape
    n = np.linspace(1, steps, steps)

    # For arrays that contain NaNs
    mask = MSD > -10000
    cutoff = steps*np.average(mask, axis=0)

    Ds = np.zeros((steps, N))
    alphas = np.zeros((steps, N))
    for i in range(N):
        for frame in range(10, steps, skips):
            cut = int(cutoff[i])
            if cut < frame:
                frame = cut
            ps, _ = curve_fit(anom, n[:frame], MSD[:frame, i], p0=[6,1])
            Ds[frame, i] = ps[0]
            alphas[frame, i] = ps[1]

    return Ds, alphas


def asym(x, y):
    # Calculates three asymmetry features based on the eigenvectors
    # of the radius of gyration tensor for a single trajectory

    # Calculuates the eigenvalues of the radius of gyration tensor
    # This tensor is just the covariance matrix of the x, y coordinates
    n = x.shape[0]
    # Modified n for trajectories that have NaNs
    mask = x > -10000
    n = n*np.average(mask)
    x, y = x[:int(n)], y[:int(n)]

    eigs, vecs = np.linalg.eig(np.cov(x, y))

    a1 = (eigs[0]**2 - eigs[1]**2)**2/(eigs[0]**2 + eigs[1]**2)**2
    a2 = min(eigs)/max(eigs)
    a3 = -np.log(1 - 0.5*(eigs[0] - eigs[1])**2/(eigs[0] + eigs[1])**2)

    # kurtosis, which requires the eigenvectors

    xi, yi = x.reshape((-1,1)), y.reshape((-1,1))
    xy = np.concatenate((xi, yi), axis=1)

    xp = np.dot(xy, vecs[:,0])
    K = np.sum((xp - np.mean(xp))**4/np.std(xp)**4)/n

    return [a1, a2, a3], eigs, vecs, K


def asyms(x, y):

    steps, N = x.shape
    a1, a2, a3, Ks = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        a123, eigs, vecs, Ks[i] = asym(x[:, i], y[:, i])
        a1[i], a2[i], a3[i] = a123[0], a123[1], a123[2]

    return a1, a2, a3, Ks


def minBoundRect(x, y):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    n = x.shape[0]
    # Modified n for trajectories that have NaNs
    mask = x > -10000
    n = n*np.average(mask)
    x, y = x[:int(n)], y[:int(n)]


    points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    # Calculate the length of the sides of the min bound rect:
    xsq = np.diff(rval, axis=0)**2
    ls = np.sqrt(xsq[:, 0] + xsq[:, 1])

    return rval, ls[:2]


def aspectRatio(x, y):
    rs, ls = minBoundRect(x, y)
    return np.max(ls)/np.min(ls), 1 - np.min(ls)/np.max(ls)


def aspectRatios(x, y):
    steps, N = x.shape
    ars = np.zeros(N)
    elongs = np.zeros(N)

    for i in range(N):
        ars[i], elongs[i] = aspectRatio(x[:, i], y[:, i])

    return ars, elongs


def bound(x, y, M):
    # I took dt out of the equation, as you both multiply and
    # divide by it, so it cancels out
    # boundedness
    N = M.shape[0]
    D = (M[1] - M[0])/(4)
    r = np.sqrt(np.max(np.diff(x)**2 + np.diff(y)**2))/2

    #trappedness
    bd = D*N/r**2
    return bd, 1 - np.exp(0.2048 - 0.25117*bd)


def bounds(xs, ys, Ms):
    # boundedness
    n, _ = Ms.shape
    # Modified for arrays that have NaNs
    mask = Ms > -10000
    cutoff = n*np.average(mask, axis=0)
    n = cutoff
    Ds = (Ms[1, :] - Ms[0, :])/4
    rs = np.sqrt(np.nanmax(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2, axis=0))/2

    #trappedness
    bds = Ds*n/rs**2
    return bds, 1 - np.exp(0.2048 - 0.25117*bds)


def efficiency(x, y):
    netD = (x[-1] - x[0])**2 + (y[-1] - y[0])**2
    total = np.sum(np.diff(x)**2 + np.diff(y)**2)
    return netD/total


def efficiencies(xs, ys):
    # Originally used x[-1] and y[-1] to get last elements, but had to do
    # something different for arrays with NaNs
    #xlast, ylast = xs[-1, :], ys[-1, :]
    n, _ = xs.shape
    # Modified for arrays that have NaNs
    mask = xs > -10000
    cutoff = n*np.average(mask, axis=0)
    n = cutoff.astype(np.int32) - 1
    xlast, ylast = np.diag(xs[n, :]), np.diag(ys[n, :])

    netD = (xlast - xs[0, :])**2 + (ylast - ys[0, :])**2
    total = np.nansum(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2, axis=0)
    return netD/total


def straight(x, y):
    netD = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    total = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    return netD/total


def straights(xs, ys):
    # Originally used x[-1] and y[-1] to get last elements, but had to do
    # something different for arrays with NaNs
    #xlast, ylast = xs[-1, :], ys[-1, :]
    n, _ = xs.shape
    # Modified for arrays that have NaNs
    mask = xs > -10000
    cutoff = n*np.average(mask, axis=0)
    n = cutoff.astype(np.int32) - 1
    xlast, ylast = np.diag(xs[n, :]), np.diag(ys[n, :])

    netD = np.sqrt((xlast - xs[0, :])**2 + (ylast - ys[0, :])**2)
    total = np.nansum(np.sqrt(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2), axis=0)
    return netD/total


def fractDim(x, y):
    n = x.shape[0] - 1
    total = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)) # from straight
    d = np.sqrt(np.max(np.diff(x)**2 + np.diff(y)**2)) # from bound

    return np.log(n)/np.log(n*n*d/total)


def fractDims(xs, ys):
    n, _ = xs.shape
    mask = xs > -10000
    cutoff = n*np.average(mask, axis=0)
    n = cutoff.astype(np.int32) - 1

    total = np.nansum(np.sqrt(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2), axis=0) # from straight
    d = np.sqrt(np.nanmax(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2, axis=0)) # from bound

    return np.log(n)/np.log(n*n*d/total)


def msdRatio(M):
    n = M.shape[0]
    ns = np.linspace(1,n,n)

    n2y, n2x = np.meshgrid(ns, ns)
    M2y, M2x = np.meshgrid(M, M)

    Mratio = M2x/M2y - n2x/n2y
    Mratio[n2x > n2y] = 0
    return MRatio


def msdRatios(Ms):
    n, N = Ms.shape
    ns = np.linspace(1,n,n)


    n2y, n2x = np.meshgrid(ns, ns)
    n2y, n2x = np.tile(n2y.reshape((n,n,1)), (1,1,N)), np.tile(n2x.reshape((n,n,1)), (1,1,N))

    M2y, M2x = np.zeros((n,n,N)), np.zeros((n,n,N))
    for i in range(N):
        M2y[:,:,i], M2x[:,:,i] = np.meshgrid(Ms[:,i], Ms[:,i])

    Mratio = M2x/M2y - n2x/n2y
    Mratio[n2x > n2y] = 0
    return Mratio


def calculateFeatures(xs, ys, dt, labelled=None, binned=None):
    Ms, Gs = msds.trajsMSD(xs, ys)
    fts = {}
    fts['x'] = np.nanmean(xs, axis=0)
    fts['y'] = np.nanmean(ys, axis=0)
    fts['Diffusion Coefficient'], fts['Alpha'] = anomModelN(Ms, dt)
    fts['Asymmetry1'], fts['Asymmetry2'], fts['Asymmetry3'], fts['Kurtosis'] = asyms(xs, ys)
    fts['Aspect Ratio'], fts['Elongation'] = aspectRatios(xs, ys)
    fts['Boundedness'], fts['Trappedness'] = bounds(xs, ys, Ms)
    fts['Efficiency'], fts['Straightness'] = efficiencies(xs, ys), straights(xs, ys)
    fts['Fractal Dimension'] = fractDims(xs, ys)
    mrs = msdRatios(Ms)
    fts['MSD Ratio'] = mrs[2, 20, :]
    fts['Gaussianity'] = Gs[10, :]

    N = fts['Alpha'].shape[0]
    if labelled:
        fts['Label'] = labelled*np.ones(N)

    fts = pd.DataFrame(fts)
    if binned:
        bins = binned[0]
        mFts, sFts = binFeatures(fts, bins)
        return Ms, fts, mFts, sFts
    else:
        return Ms, fts


def binFeatures(ft, bins):
    n = bins.shape[0] - 1
    labels = np.linspace(0, n-1, n)

    xl, yl = pd.cut(ft['x'], bins, labels=labels), pd.cut(ft['y'], bins, labels=labels)
    xl, yl = xl.to_numpy(), yl.to_numpy()
    labels = n*xl + yl
    ft['Group'] = labels

    mFts, sFts = ft.groupby(by='Group').mean(), ft.groupby(by='Group').std()
    return mFts, sFts
