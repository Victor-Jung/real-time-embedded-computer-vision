#############################################
# Object detection - YOLO - OpenCV
# Author : 
# Website : 
############################################


import cv2
import argparse
import numpy as np
from threading import Lock, Thread

################# Classes in the model ################# 
classes = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


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
        self.indices = []
        self.boxes = []
        self.classes = None
        self.class_ids = []
        self.confidences = []

    def updateClasses(self, classes) :
        self.classes = classes

    def updateNet(self, net) :
        self.model = model
    
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
            self.model.setInput(cv2.dnn.blobFromImage(cv2.resize(self.frame, (300,300)), size=(300, 300), swapRB=True))
            outs = self.model.forward()

            #Temporary variables
            indices = []
            boxes = []
            class_ids = []
            confidences = []
            
            conf_threshold = 0.5
            nms_threshold = 0.4

            for detection in outs[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .5:
                    class_id = detection[1]
                    class_name=id_class_name(class_id,classes)
                    box_x = detection[3] * Width
                    box_y = detection[4] * Height
                    box_width = detection[5] * Width
                    box_height = detection[6] * Height

                    boxes.append([box_x, box_y, box_width, box_height])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

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

#get a color for all type of classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#load specific configs
model = cv2.dnn.readNetFromTensorflow('SSD_MobileNet2/frozen_inference_graph.pb', 'SSD_MobileNet2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

#MAIN
cap = cv2.VideoCapture(0)

cv2.namedWindow("Object Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

indices = []
boxes = []
class_ids = []
confidences = []


recoProc = Processor()          #creation of the recognition opbject
recoProc.updateClasses(classes) #pass parameters variables
recoProc.updateNet(model)
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
    cv2.imshow("Object Detection", frame)
    
    # Quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

recoProc.stop() #stop the thread properly
cap.release()
cv2.destroyAllWindows()
