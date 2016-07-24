import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import time
import pdb
    

def bigblobKmeans(frame, fgmask, n_clusters):
    colorlist = [np.array([0,255,255]),np.array([255,0,255]),np.array([255,255,0]),np.array([0,0,255]),np.array([255,0,0]),np.array([0,255,0])]
    
    centroidList = []
    temp_X = np.where(fgmask!=0)
    X = np.vstack((temp_X[0],temp_X[1])).T    
    y_predict = KMeans(n_clusters = n_clusters).fit_predict(X)

    cluster_img = np.zeros(frame.shape)

    for ii in range(X.shape[0]):
        cluster_img[X[ii,0],X[ii,1],:] = colorlist[y_predict[ii]]
    # cv2.imshow('',cluster_img)
    for cls in range(n_clusters):
        centroidList.append( (np.mean( (X[:,1])[y_predict==cls]), np.mean( (X[:,0])[y_predict==cls])) )

    return centroidList


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




def fitCenters(centerList,dist, peopleCount):
    """fit a piece-linear line to stored centers"""
    """delete already fitted"""
    upward, downward= 0, 0 

    """polynomial fitting??"""
    # order = 3
    # p3 = [] #polynomial coefficients, order 3
    
    # centerListX = np.array(centerList)[:,0]
    # centerListY = np.array(centerList)[:,1]
    # sortedcenterListY = np.sort(centerListY)
    # turnInd = np.where(np.abs(centerListY[1:]-centerListY[:-1])>output_height/4)[0]
    # # turnInd = np.where(np.abs(sortedcenterListY[1:]-sortedcenterListY[:-1])>output_height/4)[0]
    # pdb.set_trace()
    # # if len(turnInd)>1:
    # #     for tt in range(len(turInd)):
    # #         subCenterList = centerList[0:turnInd[tt]]

    # # for kk in centerList:
    # #     p3.append(np.polyfit(centerListX, centerListY, order))
    # plt.figure()
    # plt.scatter(centerListX,centerListY)

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


