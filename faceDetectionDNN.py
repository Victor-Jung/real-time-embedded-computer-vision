#############################################
# Object detection - YOLO - OpenCV
# Author : 
# Website : 
############################################

from __future__ import division
import cv2
import time
import sys
import argparse
import numpy as np
from threading import Lock, Thread



def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = id_class_name(class_id, classes)

    color = (0, 0, 255)

    startPoint = (x,y)
    endPoint = (x_plus_w ,y_plus_h)

    cv2.rectangle(img, startPoint, endPoint, color, 2)

    cv2.putText(img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#Recognition processor Thread
class Processor(Thread):
    """
    Envoi en boucle sur protocole udp
    """
    def __init__(self):
        Thread.__init__(self)   #init thread
        self.lock = Lock()      #Synchronisation object
        
        self.running = True

        self.frame = np.array([[[0, 0, 0]]], dtype='uint8') #equivalent to 1x1 image

        # One big difference with the other script is the line below, here we return the image so we need to  initialze it here ! 
        self.frameOpencvDnn = np.array([[[0, 0, 0]]], dtype='uint8')
        self.bboxes = []

    def updateNet(self, net) :
        self.model = net
    
    def sendFrame(self, frame) :
        self.frame = frame

    def getResults(self) :
        #access to shared variables, verify and lock all other access
        self.lock.acquire() #lock all others lock.acquire
        
        #load in temporary variables, to allow the return
        frameOpencvDnn, bboxes = self.frameOpencvDnn, self.bboxes

        self.lock.release() #release lock
        
        return frameOpencvDnn, bboxes
        
    def run(self):
        print(("Launching recognition processor\n"), end='')
        while self.running :
            
            #Computing
            #No need to lock the "frame" access, it is called only once per iteration

            conf_threshold = 0.7

            frameOpencvDnn = self.frame.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

            self.model.setInput(blob)
            detections = self.model.forward()
            bboxes = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

            #access to shared variables, verify and lock all other access
            self.lock.acquire() #lock all others lock.acquire

            # pass temporary to atributes
            self.frameOpencvDnn = frameOpencvDnn
            self.bboxes = bboxes
            
            self.lock.release() #release lock
            
        print(("Recognition processor stopped\n"), end='')

    def stop(self) :
        self.running = False


#PRELOAD
print(cv2.__version__)

#load specific configs
modelFile = "res10FaceDetection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "res10FaceDetection/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

conf_threshold = 0.7

#MAIN
cap = cv2.VideoCapture(0)

cv2.namedWindow("Face Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

ret, frame = cap.read()
vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(0).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

indices = []
boxes = []
class_ids = []
confidences = []


recoProc = Processor()          #creation of the recognition opbject
recoProc.updateNet(net)
recoProc.start()                #calls "run" function as a thread

while True :
    #Video Stream
    ret, frame = cap.read()
    
    recoProc.sendFrame(frame) #no lag here as no lock is used

    #get results of last recognition
    #may cause a lag if the attributes are being written
    #the lag lasts as long as a memory access
    outOpencvDnn, bboxes = recoProc.getResults()

    #print((outOpencvDnn))
    fps = cap.get(cv2.CAP_PROP_FPS)
    label = "OpenCV DNN ; FPS : {:.2f}".format(fps)

    cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    #Creation of boxes overlay
    cv2.imshow("Face Detection", outOpencvDnn)

    vid_writer.write(outOpencvDnn)

    # Quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

recoProc.stop() #stop the thread properly
cap.release()
vid_writer.destroyAllWindows()
