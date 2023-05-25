import cv2
import numpy as np
import datetime
# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNetFromDarknet('yolov3_training.cfg', 'yolov3_training_final.weights')

# Define the class labels
classes = ['Scalpel']





# Load the video
cap = cv2.VideoCapture('bisturi1.mp4')

# Initialisez le détecteur d'objets
detector = cv2.SimpleBlobDetector_create()

# Set the confidence threshold
conf_threshold = 0.5

# Set the non-maximum suppression threshold
nms_threshold = 0.4
# Loop through the frames of the video
while True:
    ret, frame = cap.read()

    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Get the output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Forward pass through the network
    layer_outputs = net.forward(output_layers)

    # Initialize some variables for object detection
    boxes = []
    confidences = []
    class_ids = []

    # Loop through the outputs of each layer
    for output in layer_outputs:
        # Loop through each detection in the output
        for detection in output:
            # Extract the class probabilities and class ID
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the confidence is above the threshold
            if confidence > conf_threshold:
                # Get the center, width, and height of the bounding box
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the top-left corner of the bounding box
                left = int(center_x - width/2)
                top = int(center_y - height/2)

                # Add the bounding box, confidence, and class ID to their respective lists
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw the bounding boxes and labels on the input frame
    for i in indices:
        if i< class_ids[i]:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            if class_ids[i] < len(classes):
                label = classes[class_ids[i]]
            else:
                label = 'unknown'

            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Display the output frame
    cv2.imshow('Object detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy the window
cap.release()



