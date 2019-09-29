#############################################
# Object detection - YOLO - OpenCV
# Author : 
# Website : 
############################################


import cv2
import argparse
import numpy as np
from threading import Lock, Thread

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',
                help = 'path to yolo config file', default="yolov3.cfg")
ap.add_argument('-w', '--weights',
                help = 'path to yolo pre-trained weights', default="yolov3.weights")
ap.add_argument('-cl', '--classes',
                help = 'path to text file containing class names', default="yolov3.txt")
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

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
        self.indices = []
        self.boxes = []
        self.classes = None
        self.class_ids = []
        self.confidences = []

    def updateClasses(self, classes) :
        self.classes = classes

    def updateNet(self, net) :
        self.net = net
    
    def sendFrame(self, frame) :
        self.frame = frame

    def getResults(self) :
        #access to shared variables, verify and lock all other access
        self.lock.acquire() #lock all others lock.acquire
        
        #load in temporary variables, to allow the return
        indices, boxes, class_ids, confidences = self.indices, self.boxes, self.class_ids, self.confidences

        self.lock.release() #release lock
        
        return indices, boxes, class_ids, confidences
        
    def  run(self):
        print(("Launching recognition processor\n"), end='')
        while self.running :
            Width = self.frame.shape[1]
            Height = self.frame.shape[0]
            scale = 0.00392
            
            #Computing
            #No need to lock the "frame" access, it is called only once per iteration
            blob = cv2.dnn.blobFromImage(self.frame, scale, (416,416), (0,0,0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(get_output_layers(net))

            #Temporary variables
            indices = []
            boxes = []
            class_ids = []
            confidences = []
            
            conf_threshold = 0.5
            nms_threshold = 0.4

            #Result compilation in temporary variables
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            #access to shared variables, verify and lock all other access
            self.lock.acquire() #lock all others lock.acquire

            # pass temporary to atributes
            self.indices = indices
            self.boxes = boxes
            self.class_ids = class_ids
            self.confidences = confidences
            
            self.lock.release() #release lock
            
        print(("Recognition processor stopped\n"), end='')

    def stop(self) :
        self.running = False


#PRELOAD

#load classes to compute with
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#get a color for all type of classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#load specific configs
net = cv2.dnn.readNet(args.weights, args.config)


#MAIN
cap = cv2.VideoCapture(0)

indices = []
boxes = []
class_ids = []
confidences = []


recoProc = Processor()          #creation of the recognition opbject
recoProc.updateClasses(classes) #pass parameters variables
recoProc.updateNet(net)
recoProc.start()                #calls "run" function as a thread


while True :
    #Video Stream
    ret, frame = cap.read()
    recoProc.sendFrame(frame) #no lag here as no lock is used

    #get results of last recognition
    #may cause a lag if the attributes are being written
    #the lag lasts as long as a memory access
    indices, boxes, class_ids, confidences = recoProc.getResults()

    #Creation of boxes overlay
    for i in indices:
        #create the box
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        #render the box
        draw_prediction(frame, class_ids[i], confidences[i], int(x), int(y), int(x+w), int(y+h))

    #show the final frame (computed or not)
    cv2.imshow("object detection", frame)
    
    # Quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

recoProc.stop() #stop the thread properly
cap.release()
cv2.destroyAllWindows()
