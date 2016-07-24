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


def readBuffer(startOffset):
    for ii in range(startOffset):
        ret, frame = cap.read()
    return cap



def getFrame(frameInd):
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




if __name__ == '__main__':
        
    plt.ion()
    """input data"""
    # 191334-vv-1, 190645-vv-1
    # cap = cv2.VideoCapture('../../data/191334-vv-1.avi')
    cap = cv2.VideoCapture('../../data/192.168.31.138_01_20160706191100879.mp4')
    # cap = cv2.VideoCapture('/Users/Chenge/Downloads/2016-07-21/3-2.8mm/192.168.1.147_01_20160721171223357.mp4')
    # cap = cv2.VideoCapture('/Users/Chenge/Downloads/2016-07-21/3-4mm/192.168.1.145_01_20160721164044307.mp4')
    # cap = cv2.VideoCapture('/Users/Chenge/Downloads/2016-07-21/3-4mm/192.168.1.145_01_20160721164209992.mp4')
    # cap = cv2.VideoCapture('/Users/Chenge/Desktop/stereo_vision/peopleCounter/data/190645-vv-1.avi')

    ret, frame = cap.read()

    # imgLstBuf =  ??? #read in data in image list format
    # bufSize  = 50
    # imgLstBuf = np.zeros((1, bufSize))

    colors = np.array([[0, 1, 0],[0, 0, 1],[1, 0, 0],[1,1,0],[1,0,1]])

    """parameters"""
    # fgbg = cv2.BackgroundSubtractorMOG2()
    fgbg = cv2.BackgroundSubtractorMOG2(history=10, varThreshold=500)

    kernelSize = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))

    detector = cv2.SimpleBlobDetector()

    scale = 0.5
    output_width  = int(frame.shape[1] * scale)
    output_height = int(frame.shape[0] * scale)
    CODE_TYPE = cv2.cv.CV_FOURCC('m','p','4','v')
    # video = cv2.VideoWriter('output_detection2.avi',CODE_TYPE,6,(output_width,output_height),1)

    (Cx_old,Cy_old) = (0,0)

    centerList = {}
    peopleCount = 0
    peopleDirection = 0
    dots = []
    lines = []
    annos = []
    directionList = []
    startOffset = 377
    frameInd = 0+startOffset
    cap = readBuffer(startOffset)


    Visualize = True

    if Visualize:
        fig = plt.figure('vis')
        axL = plt.subplot(1,1,1)
        im  = plt.imshow(np.zeros([output_height,output_width,3]))
        plt.axis('off')

    dist = {}

    radiusList = []
    contourAreaList = []

    while(cap.isOpened()):
        frameInd+=1
        # print frameInd
        start = time.clock()

        frame = getFrame(frameInd)

        frame = cv2.resize(frame, (output_width, output_height), interpolation = cv2.INTER_CUBIC)
        fgmask = fgbg.apply(frame)
        ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) # THRESH_BINARY, THRESH_TOZERO
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('fgmask',fgmask)
        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = None

        """delete acient lists"""
        if len(centerList)>0:
            for jj in centerList.keys():
                if np.abs(centerList[jj][-1][2]-frameInd)>10:
                    del centerList[jj]
                    del dist[jj]

        """reference grid"""
        # lowerH, upperH = output_height/4, 3*output_height/4
        # lowerH, upperH = 2*output_height/5, 3*output_height/5
        lowerH, upperH = 9*output_height/20, 11*output_height/20

        if Visualize:
            cv2.line(frame, (0,lowerH), (output_width,lowerH), (255, 0, 255),1)
            cv2.line(frame, (0,output_height/2), (output_width,output_height/2), (255, 0, 255),1)
            cv2.line(frame, (0,upperH), (output_width,upperH), (255, 0, 255),1)

        # cntCount=0
        for cnt in contours:
            if cv2.contourArea(cnt) < 5000:
                continue
            # cntCount+=1
            # print 'cnt', cntCount
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            radiusList.append(radius)
            contourAreaList.append(cv2.contourArea(cnt))
            

            if radius > 160: # big blob
                # n_clusters = np.int(cv2.contourArea(cnt)/12500)
                n_clusters = np.int(cv2.contourArea(cnt)/13000)
                centroidList = bigblobKmeans(frame, fgmask, n_clusters)
                for nn in range(n_clusters):
                    center = (int(centroidList[nn][0]), int(centroidList[nn][1]), frameInd)
                    centerList, dist = saveCenter(center, centerList, dist)

            else: # single blob
                # ellipse = cv2.fitEllipse(cnt)
                M = cv2.moments(cnt)
                ## (x,y,time)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), frameInd)
                centerList, dist = saveCenter(center, centerList, dist)

            if Visualize:
                try:
                    # axL.lines.pop(0).remove()  # can only remove the 1st one
                    if len(lines)>0:
                        for ll in range(len(lines)):
                            axL.lines.pop(ll)
                        lines = []
                    fig.canvas.draw()
                except:
                    line_exist = 0
                plt.show()

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # cv2.ellipse(frame,ellipse,(0,255,255),2)
                
                cv2.circle(frame, center[:-1], 5, (0, 0, 255), -1)
                # if (Cx_old!=0) and (Cy_old!=0):
                #     cv2.line(frame, center, (Cx_old,Cy_old), (0, 255, 0), 1)
                # (Cx_old,Cy_old) = center
                im.set_data(frame[:,:,::-1])
                if len(centerList)>0:
                    for cc in centerList.keys():
                        # if np.min(np.array(centerList[cc])[:,1])>lowerH or np.max(np.array(centerList[cc])[:,1])<upperH:
                        #     continue #don't draw the points outside of detection region
                        # else:
                        #     lines.append(axL.plot(np.array(centerList[cc])[:,0],np.array(centerList[cc])[:,1],'-o',color = np.array(colors[np.mod(cc,5)]), linewidth=1))
                    
                        lines.append(axL.plot(np.array(centerList[cc])[:,0],np.array(centerList[cc])[:,1],'-o',color = np.array(colors[np.mod(cc,5)]), linewidth=1))
                    line_exist = 1
                    fig.canvas.draw()

        temp = peopleCount
        if len(centerList)>0:
            peopleCount, directionList, dist, centerList= fitCenters(centerList, dist, peopleCount)

        if Visualize:
            im.set_data(frame[:,:,::-1])
            fig.canvas.draw()
            if temp!=peopleCount:
                if len(annos)>0:
                    for aa in range(len(annos)):    
                        annos[aa].remove()  
                    annos = []
                    fig.canvas.draw()
                if len(directionList)>0:
                    annos.append(plt.annotate('upward = '+str(np.sum(np.array(directionList)==2)), xy=(0.06*output_width,0.8*output_height), color='#ee8d18'))
                    annos.append(plt.annotate('downward = '+str(np.sum(np.array(directionList)==1)), xy=(0.06*output_width,0.85*output_height),color='#ee8d18'))
                    annos.append(plt.annotate('others = '+str(np.sum(np.array(directionList)==3)), xy=(0.06*output_width,0.9*output_height),color='red'))
                    annos.append(plt.annotate('total = '+str(peopleCount), xy=(0.06*output_width,0.95*output_height),color='#ee8d18'))

                    fig.canvas.draw()


            # if frameInd>300:
            fname = '../frames/'+str(frameInd).zfill(6)+'.jpg'
            plt.savefig(fname)
            # # cv2.imwrite(fname,fgmask)


        # Detect blobs.
        # keypoints = detector.detect(fgmask)
        # print len(keypoints)
        # fgmask_keypoint = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('fgmask',fgmask)
        # end = time.clock()
        # print('fps: {}'.format(1 / (end - start)))

        # cv2.imshow('frame',frame)
        # video.write(frame)


        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        end = time.clock()
        print 'time:', (end - start)


    cap.release()
    video.release()
    cv2.destroyAllWindows()
