import cv2
import time
import pdb
import numpy as np
from sklearn.cluster import KMeans 


def readBuffer(startOffset, cap):
    for ii in range(startOffset):
        ret, frame = cap.read()
    return cap


def getFrame(cap, frameInd):
    ret, frame = cap.read()
    if ret == False:
        return None
    # frame = imgLstBuf[np.mod(frameInd,bufSize)]
    return frame


def bigblobKmeans(frame, fgmask, n_clusters):
    colorlist = [np.array([0,255,255]),np.array([255,0,255]),np.array([255,255,0]),np.array([0,0,255]),np.array([255,0,0]),np.array([0,255,0])]
    centroidList = []
    temp_X = np.where(fgmask!=0)
    X = np.vstack((temp_X[0],temp_X[1])).T    
    y_predict = KMeans(n_clusters = n_clusters,n_init=3,max_iter=150,tol=1).fit_predict(X)

    # cluster_img = np.zeros(frame.shape)
    # for ii in range(X.shape[0]):
    #     cluster_img[X[ii,0],X[ii,1],:] = colorlist[y_predict[ii]]
    # cv2.imshow('',cluster_img)
    # cv2.waitKey(0)
    for cls in range(n_clusters):
        centroidList.append( (np.mean( (X[:,1])[y_predict==cls]), np.mean( (X[:,0])[y_predict==cls])) )

    return centroidList


def getBlobRatio(blobmask, validTrackUpperBound, validTrackLowerBound):
    """estimate number of people passing just by horizontal foreground pixel ratio"""
    """get ratio for EACH blob"""    
    horizRatio = np.dot(np.sum(blobmask[validTrackUpperBound: validTrackLowerBound,:]!=0,1), 1.0/blobmask.shape[1])
    peak = np.max(horizRatio)
    if peak > 0.1:
        peakLoc = np.where(horizRatio==peak)[0]
        peakLoc = np.mean(peakLoc)
    else:
        peakLoc = None
    return (peak, peakLoc)


def smooth(xk, yk):
    from scipy.interpolate import interp1d
    f1 = interp1d(xk, yk, kind='linear', axis=-1, copy=True, bounds_error=True, fill_value=np.nan, assume_sorted=False)
    x_smooth_per_pixel = np.arange(xk.min(), xk.max(),0.5)
    y_smooth_per_pixel = f1(x_smooth_per_pixel)
    
    # x_smooth_same_len = np.linspace(x_smooth_per_pixel.min(), x_smooth_per_pixel.max(),len(xk))
    # f2 = interp1d(x_smooth_per_pixel, y_smooth_per_pixel, kind='slinear', axis=-1, copy=True, bounds_error=True, fill_value=np.nan, assume_sorted=False)
    # y_smooth_same_len = f2(x_smooth_same_len)
    # return x_smooth_same_len, y_smooth_same_len
    return x_smooth_per_pixel, y_smooth_per_pixel




