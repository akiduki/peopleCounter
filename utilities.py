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

def mergeCenterList(centerList):
    """merge lists that are too close both in space and in time"""
    keys = centerList.keys()
    for ii1 in range(len(keys)-1):
        for ii2 in range(cc+1,len(keys),1):
            cc1 = keys[ii1]
            cc2 = keys[ii2]
            if np.sum(np.abs(np.array(centerList[cc1])[:,1] - np.array(centerList[cc2])[:,1]) + \
                np.abs(np.array(centerList[cc1])[:,0] - np.array(centerList[cc2])[:,0]))/len(centerList[cc1]) <10:
                print "merge?"
                pdb.set_trace()
    return centerList


def checkBlobSize(fgmask):
    """check blob sizes in different videos/regions"""
    """reference grid"""
    scale = 0.5
    output_width  = int(fgmask.shape[1] * scale)
    output_height = int(fgmask.shape[0] * scale)

    upperH,lowerH  = output_height/4, 3*output_height/4
    # upperH,lowerH = 2*output_height/5, 3*output_height/5    
    return (np.sum(fgmask[0:upperH,:]!=0),  np.sum(fgmask[upperH:output_height/2,:]!=0), np.sum(fgmask[output_height/2:lowerH,:]!=0),np.sum(fgmask[lowerH:output_height,:]!=0) )

def fitCenters(centerList,dist, peopleCount):
    """fit a piece-linear line to stored centers"""
    """delete already fitted"""
    upward, downward= 0, 0 
    for cc in centerList.keys():
        """estimate peopel count"""
        if (np.min(np.array(centerList[cc])[:,1])<= lowerH) and (np.max(np.array(centerList[cc])[:,1])>=upperH):
            peopleCount+=1
            print "peopleCount",peopleCount
            """estimate direction"""
            if ((np.array(centerList[cc])[:,1][1:]-np.array(centerList[cc])[:,1][:-1])>=0).sum()/float(len(centerList[cc]))>=0.70:
                peopleDirection = 1 ## downward
            elif ((np.array(centerList[cc])[:,1][1:]-np.array(centerList[cc])[:,1][:-1])<=0).sum()/float(len(centerList[cc]))>=0.70:
                peopleDirection = 2 # upward
            elif np.mean(np.array(centerList[cc])[:,1][-3:])< lowerH and np.mean(np.array(centerList[cc])[:,1][:3])>upperH:
                peopleDirection = 2 # upward
            elif np.mean(np.array(centerList[cc])[:,1][:3])< lowerH and np.mean(np.array(centerList[cc])[:,1][-3:])>upperH:
                peopleDirection = 1 # downward
            else:
                peopleDirection = 3 # odd cases
            directionList.append(peopleDirection)

            """delete previous history, don't over count, but still keep the last few for appending"""
            del centerList[cc]
            del dist[cc]
            # end = time.clock()
            # print('detection time: {}'.format(end - start))

    return peopleCount, directionList, dist, centerList


def saveCenter(center, centerList, dist):
    """save center locations"""
    """put center to the nearest list"""
    if len(centerList)==0:
        listInd = 0
        centerList[listInd] = []
        centerList[listInd].append(center)
    else:
        # if peopleCount in [0,1]:
        #     print peopleCount, centerList.keys(), dist.keys()
        for cc in centerList.keys():
            dist[cc] = []
            dist[cc] = np.sqrt((center[0]-centerList[cc][-1][0])**2+(center[1]-centerList[cc][-1][1])**2)
        smallestDist = np.min(dist.values())
        nearestCenterInd = dist.keys()[np.where(dist.values()==smallestDist)[0][0]]
        if smallestDist>output_height/5:
            """far away from all existing lists"""
            listInd = np.max(centerList.keys())
            listInd +=1
            print "new list bc too far away"
            centerList[listInd] = []
            centerList[listInd].append(center)    
        else:
            """append to the nearest list"""
            centerList[nearestCenterInd].append(center)

    return centerList, dist
