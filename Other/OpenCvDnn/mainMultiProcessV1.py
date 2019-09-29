'''' MultiThreading '''
'''
class Display(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, cap):
        Thread.__init__(self)
        self.cap = cap

    def run(self):
      print("Enter in display")
      k = 0
      while k<100 :

        k=k+1
        #print(k)

        # Capture frame-by-frame
        ret, frame = self.cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)

        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      cv2.destroyAllWindows()


class LaunchNn(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, cap):
        Thread.__init__(self)
        self.cap = cap

    def run(self):

      print("Enter in Nn")

      ### Loading model ###
      model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                            'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

      j=0

      while j<100 :

        j=j+1
        #print(j)

        ret, image = self.cap.read()

        image_height, image_width, _ = image.shape

        ### Forward Propagation
        model.setInput(cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), size=(300, 300), swapRB=True))
        output = model.forward()

        ### Display Classe Name
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .6:
                class_id = detection[1]
                class_name=id_class_name(class_id,classNames)
                print("Classe : " + class_name + " Proba : " + str(round(detection[2], 2))) 

        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
'''
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Pipe

#ray.init()

################# Classes in the model ################# 
classNames = {0: 'background',
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



''' MultiProcessing '''


def displayStream(pipeF1):

  font = cv2.FONT_HERSHEY_SIMPLEX
  bottomLeftCornerOfText = (10,200)
  fontScale = 1
  fontColor = (0,0,255)
  lineType = 2

  window_name = "window"

  cap = cv2.VideoCapture(0)   

  cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

  for k in range(0,100):
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    if k % 5 == 0:
      #q.put(frame)
      pipeF1.send(frame)
      #print(pipeF1.recv())

    # Display the resulting frame
    cv2.imshow(window_name,frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
    k=k+1

  cv2.destroyAllWindows()


def launchNeuralNetwork(pipeF2):

  for j in range(0,25):

    #image = q.get()
    image = pipeF2.recv()

    ### Loading model ###
    model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                          'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    image_height, image_width, _ = image.shape

    model.setInput(cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), size=(300, 300), swapRB=True))
    output = model.forward()

    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .6:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            #pipeF2.send(str("Classe : " + class_name + " Proba : " + str(round(detection[2], 2))))
            print("Classe : " + class_name + " Proba : " + str(round(detection[2], 2)))



################################## Main ################################## 

#q = Queue()
pipeF1, pipeF2 = Pipe()

p1 = Process(target=displayStream, args=(pipeF1,))
p1.start()
p2 = Process(target=launchNeuralNetwork, args=(pipeF2,))
p2.start()

p1.join()
p2.join()