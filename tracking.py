import cv2
import numpy as np

colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0), (0,128,128), (128,0,128), (128,128,0)]

class Blob(object):
    def __init__(self, center, minx, maxx, miny, maxy):
        self.center = center
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy


class Track(object):
    def __init__(self, direction, color):
        self.centerList = []
        self.lifetime = 0
        self.inactiveCount = 0
        self.activeCount = 0
        self.direction = direction
        self.counted = False
        self.color = color

    def updateTrack(self, blob):
        self.centerList.append(blob.center)
        self.minx = blob.minx
        self.maxx = blob.maxx
        self.miny = blob.miny
        self.maxy = blob.maxy
        self.inactiveCount = 0
        self.activeCount += 1

    def predictCenter(self):
        return self.centerList[-1]

    def plot(self, img):
        for i in reversed(xrange(len(self.centerList) - 1)):
            cv2.line(img, self.centerList[i + 1], self.centerList[i], self.color)
            cv2.circle(img, self.centerList[i + 1], 2, self.color, -1)

    def printTrack(self):
        for i in reversed(xrange(len(self.centerList) - 1)):
            print self.centerList[i],
        print ' '


class Tracking(object):
    def __init__(self, countUpperBound, countLowerBound, validTrackUpperBound, validTrackLowerBound):
        self.countUpperBound = countUpperBound
        self.countLowerBound = countLowerBound
        self.validTrackUpperBound = validTrackUpperBound
        self.validTrackLowerBound = validTrackLowerBound
        self.counter = 0


    def updateTrack(self, blobs, tracks, distThreshold, inactiveThreshold):
        # print tracks
        nBlob = len(blobs)
        nTrack = len(tracks)
        distMatrix = np.zeros((nTrack, nBlob))
        blobMark = np.zeros(nBlob)
        trackMark = np.zeros(nTrack)
        for idxBlob, blob in enumerate(blobs):
            for idxTrack, track in enumerate(tracks):
                distMatrix[idxTrack, idxBlob] = self.distBlobTrack(blob, track)

        nAssignedBlob = 0
        for idxBlob, blob in enumerate(blobs):
            minDist = 10000
            minIdxTrack = 0
            closestTrack = None
            # for each blob, find the closest track < distThreshold. 
            # if the track is not picked yet, assign blob to the track
            for idxTrack, track in enumerate(tracks):
                if distMatrix[idxTrack, idxBlob] < minDist:
                    minDist = distMatrix[idxTrack, idxBlob]
                    minIdxTrack = idxTrack
                    closestTrack = track
            if minDist < distThreshold and trackMark[minIdxTrack] == 0:
                # print minDist
                closestTrack.updateTrack(blob)
                trackMark[minIdxTrack] = 1
                blobMark[idxBlob] = 1
                nAssignedBlob += 1

        # print 'Assigned blob: %s' % nAssignedBlob

        # for not assigned blob, determine if it is a valid new track
        newTracks = []
        for idxBlob, blob in enumerate(blobs):
            if blobMark[idxBlob] == 1:
                continue
            direction = self.appearRegion(blob)
            if direction != 0:
                newTrack = Track(direction, colors[self.counter % len(colors)])
                newTrack.updateTrack(blob)
                newTracks.append(newTrack)
                self.counter += 1

        nUp = 0
        nDown = 0
        for idxTrack, track in enumerate(tracks):
            # for not assigned track, increment inactive states, delete if more than inactiveThreshold
            if trackMark[idxTrack] == 0:
                track.activeCount = 0
                track.inactiveCount += 1
                if track.inactiveCount >= inactiveThreshold:
                    trackMark[idxTrack] = 2
            # if track has passed detection region, increment up/down counter, then inactivate track
            elif not track.counted:
                if track.direction == -1 and track.centerList[-1][1] > self.countLowerBound:
                    nDown += 1
                    track.counted = True
                elif track.direction == 1 and track.centerList[-1][1] < self.countUpperBound:
                    nUp += 1
                    track.counted = True

        tracks = [tracks[i] for i in range(len(tracks)) if trackMark[i] <= 1]

        # append new tracks to track list
        tracks.extend(newTracks)
        return (tracks, nUp, nDown)


    def distBlobTrack(self, blob, track):
        predictCenter = track.predictCenter()
        dist = np.linalg.norm(np.array(blob.center) - np.array(predictCenter))
        # print blob.center, predictCenter, dist
        return dist


    def appearRegion(self, blob):
        if blob.center[1] < self.validTrackUpperBound:
            return -1
        elif blob.center[1] > self.validTrackLowerBound:
            return 1
        else:
            return 0

