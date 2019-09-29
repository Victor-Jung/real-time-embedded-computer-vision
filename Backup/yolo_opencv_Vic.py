import cv2
import argparse
import numpy as np
from imutils.video import FPS
from threading import Thread
import imutils
import queue
import datetime

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

def process(qFrame) :
    '''global indices
    global boxes
    global classes
    global class_ids
    global confidences'''
    
    while True :

        frame = qFrame.get()

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

        ### Fill the queues
        qIndices.put(indices)
        qBoxes.put(boxes)
        qClass_ids.put(class_ids)
        qConfidences.put(confidences)

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
 
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
 
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
 
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
 
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
 
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()



class WebcamVideoStream:

    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
 
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return self.frame
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


############### Preload ###############
'''global indices
global boxes
global classes
global class_ids
global confidences'''

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

############### Main ###############

indices = []
boxes = []
class_ids = []
confidences = []

qFrame = queue.Queue()
qIndices = queue.Queue()
qBoxes = queue.Queue()
qClass_ids = queue.Queue()
qConfidences = queue.Queue()

#cap = cv2.VideoCapture(0)
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
frame = vs.read()

threadProcess = Thread(target=process, args=(qFrame,))
threadProcess.start()

## Number of frame after going through the neural network
count = 1;

### Video Stream ###
while True :
    
    frame = vs.read()
    fps.update()

    frame = imutils.resize(frame, width=550)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    if count % 100 == 0:

        qFrame.put(frame)

        indices = qIndices.get()
        boxes = qBoxes.get()
        class_ids = qClass_ids.get()
        confidences = qConfidences.get()

        count = 0
        print('Tic')

    count += 1

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], int(x), int(y), int(x+w), int(y+h))

    cv2.imshow("object detection", frame)
    cv2.waitKey(1)
    fps.update()

    # Quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
