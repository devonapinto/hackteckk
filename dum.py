import cv2
import numpy as np

# Load the pre-trained deep learning model for iris detection
net = cv2.dnn.readNet('iris_detection_model.weights', 'iris_detection_model.cfg')

# Load the input image
image = cv2.imread("C:\\Users\\maqwi\\Desktop\\hack\\archive (1)\\processed_images\\train\\cataract\\image_3.png")

# Resize the image to a fixed size (e.g., 300x300) and preprocess it
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

# Set the input to the neural network
net.setInput(blob)

# Forward pass through the network to detect irises
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # If the confidence is above a certain threshold, draw the iris region
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the output image with iris detection
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
