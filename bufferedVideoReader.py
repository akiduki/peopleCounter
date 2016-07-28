# Buffered thread-safe video reader class for people counting
import cv2
import numpy as np
import imageio

from datetime import datetime
import time

import Queue
from threading import Thread

BufFrameQ = Queue.Queue()
TStampQ = Queue.Queue()

framerate = 15

def BufVideoReader(url, frame_q, ts_q, framerate):
    # url is the RTSP link, q is the FIFO queue for storing video frames
    cap = cv2.VideoCapture(url)

    while True:
        try:
            ret, frm = cap.read()
            frame_q.put(frm)
            ts_q.put(time.time())
            # wait until next time interval
            time.sleep(1./framerate)
        except KeyboardInterrupt:
            break

def _test():
    while True:
        try:
            if not BufFrameQ.empty():
                frm = BufFrameQ.get()
                ts = TStampQ.get()
                filename = datetime.fromtimestamp(ts).strftime('%m-%d_%H:%M:%S.%f') + '.jpg'
                imageio.imwrite(filename, frm)
                # print out log
                print 'Saved image', filename
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    url = 'rtsp://admin:WikkitLabs@192.168.31.129:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1'

    print 'Spawn a daemon thread for fetching frames to a list'
    worker = Thread(target=BufVideoReader, args=(url, BufFrameQ, TStampQ, framerate, ))
    worker.setDaemon(True)
    worker.start()

    print 'Now run the test routine to save the frames to images'
    _test()

