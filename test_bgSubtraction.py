import numpy as np
import cv2
import time
# import SimpleCV
import pdb
import matplotlib.pyplot as plt

# 191334-vv-1, 190645-vv-1
# cap = cv2.VideoCapture('../../data/191334-vv-1.avi')
# cap = cv2.VideoCapture('../../data/192.168.31.138_01_20160706191100879.mp4')
cap = cv2.VideoCapture('/Users/Chenge/Desktop/stereo_vision/peopleCounter/data/190645-vv-1.avi')

ret, frame = cap.read()

# fgbg = cv2.BackgroundSubtractorMOG2()
fgbg = cv2.BackgroundSubtractorMOG2(history=10, varThreshold=500)

kernelSize = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))

detector = cv2.SimpleBlobDetector()

scale = 0.5
output_width  = int(frame.shape[1] * scale)
output_height = int(frame.shape[0] * scale)
CODE_TYPE = cv2.cv.CV_FOURCC('m','p','4','v')
video = cv2.VideoWriter('output_detection2.avi',CODE_TYPE,6,(output_width,output_height),1)

(Cx_old,Cy_old) = (0,0)

centerList = []
peopleCount = 0
peopleDirection = 0
dots = []
annos = []
directionList = []
frameInd = 0
Visualize = False


if Visualize:
    fig = plt.figure('vis')
    axL = plt.subplot(1,1,1)
    im  = plt.imshow(np.zeros([output_height,output_width,3]))
    plt.axis('off')


while(cap.isOpened()):
    frameInd+=1
    ret, frame = cap.read()
    if ret == False:
        break
    print frameInd
    if frameInd ==70:
        start = time.clock()

    frame = cv2.resize(frame, (output_width, output_height), interpolation = cv2.INTER_CUBIC)
    fgmask = fgbg.apply(frame)
    ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) # THRESH_BINARY, THRESH_TOZERO
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None

    """reference grid"""
    if Visualize:
        cv2.line(frame, (0,output_height/4), (output_width,output_height/4), (255, 0, 255),1)
        cv2.line(frame, (0,output_height/2), (output_width,output_height/2), (255, 0, 255),1)
        cv2.line(frame, (0,3*output_height/4), (output_width,3*output_height/4), (255, 0, 255),1)

    cntCount=0
    for cnt in contours:
        if cv2.contourArea(cnt) < 5000:
            continue
        # cntCount+=1
        # print 'cnt', cntCount
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # ellipse = cv2.fitEllipse(cnt)
        M = cv2.moments(cnt)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        """save center locations"""
        centerList.append(center)
        # if radius > 50:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # cv2.ellipse(frame,ellipse,(0,255,255),2)
        
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # if (Cx_old!=0) and (Cy_old!=0):
        #     cv2.line(frame, center, (Cx_old,Cy_old), (0, 255, 0), 1)
        # (Cx_old,Cy_old) = center

        if Visualize:
            if len(centerList)>0:
                lines = axL.plot(np.array(centerList)[:,0],np.array(centerList)[:,1],'-o',color = np.array([0, 1, 0]), linewidth=1)
                line_exist = 1
                # dots.append(axL.scatter(np.array(centerList)[:,0],np.array(centerList)[:,1]))

    if len(centerList)>2:
        if (np.min(np.array(centerList)[:,1])< output_height/4) and (np.max(np.array(centerList)[:,1])>3*output_height/4):
            peopleCount+=1
            print "peopleCount",peopleCount
            if ((np.array(centerList)[:,1][1:]-np.array(centerList)[:,1][:-1])>=0).sum()/float(len(centerList))>=0.70:
                peopleDirection = 1 ## downward
            elif ((np.array(centerList)[:,1][1:]-np.array(centerList)[:,1][:-1])<=0).sum()/float(len(centerList))>=0.70:
                peopleDirection = 2 # upward
            elif np.array(centerList)[:,1][-1]< output_height/4 and np.array(centerList)[:,1][0]>3*output_height/4:
                peopleDirection = 2 # upward
            elif np.array(centerList)[:,1][0]< output_height/4 and np.array(centerList)[:,1][-1]>3*output_height/4:
                peopleDirection = 1 # downward

            directionList.append(peopleDirection)
            centerList = []
            if len(annos)>0:
                for aa in range(len(annos)):
                    annos[aa].remove()  
                annos = []
            fig.canvas.draw()
            annos.append(plt.annotate('upward = '+str(np.sum(np.array(directionList)==2)), xy=(0.06*output_width,0.9*output_height), color='#ee8d18'))
            annos.append(plt.annotate('downward = '+str(np.sum(np.array(directionList)==1)), xy=(0.06*output_width,0.95*output_height),color='#ee8d18'))
            fig.canvas.draw()
            end = time.clock()
            print('detection time: {}'.format(end - start))


    # fname = '../frames/'+str(frameInd).zfill(6)+'.jpg'
    # plt.savefig(fname)
    if Visualize:
        im.set_data(frame[:,:,::-1])
        fig.canvas.draw()
        plt.show()
        try:
            axL.lines.pop(0)
        except:
            line_exist = 0


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

cap.release()
video.release()
cv2.destroyAllWindows()
