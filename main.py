import math

from ultralytics import YOLO
import cv2
import cvzone

from sort import *

cap = cv2.VideoCapture("cars.mp4")

cap.set(3,1280)
cap.set(4,720)

model = YOLO("YOLO Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")


#Tracker

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

line_coords = [400, 297, 673, 297]

total_vehicles = 0

total_count = []


while True:
    success, img = cap.read()

    #creating an image region to detect vehicles going a certain direction
    img_region = cv2.bitwise_and(img, mask)


    results = model(img_region, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes


        #creting bounding boxes for objects
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2,y2), (0,200,0), 3)


            #adding confidence value for objects detected
            conf = math.ceil((box.conf[0]*100))/100

            #adding class names for building boxes
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            #if the object detected within the mask is a car, truck, motorbike or bus with a confidence level of 0.4 or more, the object is counted.
            if currentClass == "car" or currentClass == "truck" or currentClass == "motorbike" or currentClass == "bus" and conf > 0.4:

                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
                currentArray = np.array([x1,y1,x2,y2,conf])

                detections = np.vstack((detections, currentArray))


    #adds tracking ID for each bounding box
    tracking_results = tracker.update(detections)

    #creating a line so when each car crosses the line, it is counted
    cv2.line(img, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (0, 0, 255), 5)



    for coords in tracking_results:
        x1, y1, x2, y2, id = coords
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        #finding centre of each bounding box
        cx, cy = x1+(x2-x1)//2, y1+(y2-y1)//2


        if line_coords[0]<cx<line_coords[2] and line_coords[1]-15<cy<line_coords[3]+15:


            #since we added the id associated with every vehicle to "total_count", we can check if the car has already been counted in the next frame to prevent the same car being counted twice
            if total_count.count(id) == 0:
                total_count.append(id)

    #displays the total ammount of cars crossing the road
    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50, 50))


    cv2.imshow("Image", img)
    cv2.waitKey(1)

