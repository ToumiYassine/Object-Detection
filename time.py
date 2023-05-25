import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = ["Scalpel"]


cap = cv2.VideoCapture('final5.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))


appearance_times = []
object_detected = False

# Define start and end times of object appearance
start_time = None
end_time = None
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_names)


    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_time not in appearance_times:
                    appearance_times.append(current_time)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    duration = appearance_times[-1] - appearance_times[0]
    print(f"Object appeared for {duration} seconds")
    print(f"object appeared at {appearance_times[0]} seconds")
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == 27:

        break




cap.release()

cv2.destroyAllWindows()