#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
from threading import Thread

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
        

#PRELOAD
global frame
global indices
global boxes
global classes
global class_ids
global confidences

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

#MAIN

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

indices = []
boxes = []
class_ids = []
confidences = []

def process() :
    global frame
    global indices
    global boxes
    global classes
    global class_ids
    global confidences
    
    while True :
        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392
        
        #Analyse
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

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

threadProcess = Thread(target=process)
threadProcess.start()

while True :
    #Video Stream
    ret, frame = cap.read()

    #Rendu
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], int(x), int(y), int(x+w), int(y+h))

    cv2.imshow("object detection", frame)
    # Quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
