import numpy as np
import cv2
import time
import math
from tracking import Tracking
from tracking import Blob

from ConfigParser import SafeConfigParser

from utilities import bigblobKmeans, getBlobRatio
from utilities import readBuffer, getFrame
import Queue
from bufferedVideoReader import BufVideoReader
from threading import Thread

import pdb

configfile = './parameters.ini'

def json_dump(totalDown, totalUp, timestamp):
    """write data into json file"""
    countingData = {
        'direction' : 1, ##1 is downward, 2 upward
        'totalDownCount' : totalDown,
        'direction' : 2, 
        'totalUpCount' : totalUp,
        'timestamp' : timestamp,
        }

    with open('data.json', 'w') as f:
         json.dump(countingData, f)

def getFrmRSTP(BufFrameQ,TStampQ):
    try:
        if not BufFrameQ.empty():
            frm = BufFrameQ.get()
            ts = TStampQ.get()
    except KeyboardInterrupt:
        return None, None

    return frm, ts


if __name__ == '__main__':
    """input data"""
    # 191334-vv-1, 190645-vv-1
    # cap = cv2.VideoCapture('191334-vv-1.avi')
    # cap = cv2.VideoCapture('../opencv/192.168.1.145_01_20160721164209992.mp4')
    # cap = cv2.VideoCapture('../PeopleCounterLocal/01_20160721164209992.avi')
    cap = cv2.VideoCapture('/Users/Chenge/Desktop/stereo_vision/peopleCounter/data/2016-07-21/3-4mm/192.168.1.145_01_20160721164209992.mp4')
    # cap = cv2.VideoCapture('/Users/Chenge/Downloads/2016-07-20/4mm-2.65/192.168.0.100_01_20160720171945536.mp4')
    # cap = cv2.VideoCapture('/Users/yuanyi.xue/Downloads/2016-07-21/3m-4mm/192.168.1.145_01_20160721164209992.mp4')

    startOffset = 300;
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, startOffset);

    # startOffset = 300
    # cap = readBuffer(startOffset, cap)
    ret, frame = cap.read()



    """prepare for RSTP"""
    # BufFrameQ = Queue.Queue()
    # TStampQ = Queue.Queue()
    """Spawn a daemon thread for fetching frames to a list"""
    # worker = Thread(target=BufVideoReader, args=(url, BufFrameQ, TStampQ, RSTPframerate, ))
    # worker.setDaemon(True)
    # worker.start()
    
    # frame, ts = getFrmRSTP(BufFrameQ,TStampQ)


    """parameters from parameters.ini"""
    parser = SafeConfigParser()
    parser.read(configfile)
    mog2History = parser.getint('PeopleCounting', 'mog2History')
    mog2VarThrsh = parser.getint('PeopleCounting', 'mog2VarThrsh')
    mog2Shadow = parser.getboolean('PeopleCounting', 'mog2Shadow')
    mog2LearningRate = parser.getfloat('PeopleCounting', 'mog2LearningRate')
    kernelSize = parser.getint('PeopleCounting', 'kernelSize')
    scale = parser.getfloat('PeopleCounting', 'scale')
    areaThreshold = math.pi * parser.getfloat('PeopleCounting', 'areaRadius')**2
    peopleBlobSize = parser.getint('PeopleCounting', 'peopleBlobSize')
    distThreshold = parser.getint('PeopleCounting', 'distThreshold')
    countingRegion = map(int, parser.get('PeopleCounting', 'countingRegion').split(','))
    upperTrackingRegion = map(int, parser.get('PeopleCounting', 'upperTrackingRegion').split(','))
    lowerTrackingRegion = map(int, parser.get('PeopleCounting', 'lowerTrackingRegion').split(','))
    inactiveThreshold = parser.getint('PeopleCounting', 'inactiveThreshold')
    singlePersonBlobSize = parser.getint('PeopleCounting', 'singlePersonBlobSize')
    Debug = parser.getboolean('PeopleCounting', 'Debug')
    Visualize = parser.getboolean('PeopleCounting', 'Visualize') or Debug
    useRatioCriteria = parser.getboolean('PeopleCounting', 'useRatioCriteria')

    """ Initialize MOG2, VideoWriter, and tracking """
    fgbg = cv2.BackgroundSubtractorMOG2(mog2History, mog2VarThrsh, mog2Shadow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))

    output_width  = int(frame.shape[1] * scale)
    output_height = int(frame.shape[0] * scale)
    CODE_TYPE = cv2.cv.CV_FOURCC('m','p','4','v')
    video = cv2.VideoWriter('output_detection.avi',CODE_TYPE,30,(output_width,output_height*2),1)

    trackingObj = Tracking(countingRegion, upperTrackingRegion, lowerTrackingRegion, peopleBlobSize, useRatioCriteria)
    tracks = []
    totalUp = 0
    totalDown = 0
    frameInd = startOffset

    while(cap.isOpened()):
        start = time.clock()
        ret, frame = cap.read()
        """rstp"""
        # frame, ts = getFrmRSTP(cap,frameInd)
        if ret == False:
            break

        print 'Frame # %s' % frameInd
        frameInd += 1
        # frameInd = ts

        # resize image, background subtraction and post-processing of blob
        frame = cv2.resize(frame, (output_width, output_height), interpolation = cv2.INTER_CUBIC)
        fgmask = fgbg.apply(frame, mog2LearningRate)
        ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) # THRESH_BINARY, THRESH_TOZERO
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        maskedFrame = cv2.bitwise_and(frame, frame, mask = fgmask)

        # find blobs
        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = None
        blobs = []

        for cnt in contours:
            if cv2.contourArea(cnt) < areaThreshold:
                continue
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            l,u,w,h = cv2.boundingRect(cnt)
            peakVal = None
            peakLoc = None
            if useRatioCriteria:
                temp = np.zeros_like(fgmask)
                temp[u:u+h,l:l+w] =1
                blobmask = fgmask * temp
                (peakVal, peakLoc) = getBlobRatio(blobmask, countingRegion[2], countingRegion[3])

            blobs.append(Blob((int(x), int(y)), l, l + w, u, u + h, peakVal, peakLoc, frameInd))

            if Visualize:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # cv2.ellipse(frame,ellipse,(0,255,255),2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(center), center, font, 1, (0,255,255), 1)


        # tracking
        tracks, nUp, nDown = trackingObj.updateAllTrack(blobs, tracks, distThreshold, inactiveThreshold)
        totalUp += nUp
        totalDown += nDown
        print '# UP %s' % totalUp
        print '# DOWN %s' % totalDown

        # Visualize tracking region, counting region and tracks
        if Visualize:
            cv2.rectangle(frame, (countingRegion[0], countingRegion[2]), 
                            (countingRegion[1], countingRegion[3]), (0, 0, 255), 2)
            cv2.rectangle(frame, (upperTrackingRegion[0], upperTrackingRegion[2]), 
                            (upperTrackingRegion[1], upperTrackingRegion[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (lowerTrackingRegion[0], lowerTrackingRegion[2]), 
                            (lowerTrackingRegion[1], lowerTrackingRegion[3]), (255, 0, 0), 2)
            cv2.putText(frame, '# UP %s' % totalUp, (5, output_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, '# DOWN %s' % totalDown, (5, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            for idxTrack, track in enumerate(tracks):
                track.plot(frame)
                # track.printTrack()

        end = time.clock()
        print('fps: {}'.format(1 / (end - start)))

        if Visualize:
            maskedFrame = np.vstack((frame, maskedFrame))
            cv2.imshow('frame', maskedFrame)
            video.write(maskedFrame)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    cap.release()
    video.release()
    cv2.destroyAllWindows()
