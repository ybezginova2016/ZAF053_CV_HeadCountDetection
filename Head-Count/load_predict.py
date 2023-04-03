import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the model
model = keras.models.load_model('saved_model')

# Open the video file
cap = cv2.VideoCapture('test_3.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Stop the loop if we have reached the end of the video
    if not ret:
        break
    
    # Preprocess the image
    img = cv2.resize(frame, (299, 299))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Run prediction on the image
    predictions = model.predict(img)

    # Extract the count from the predictions
    count = int(predictions[0])

    # Display the count on the image
    cv2.putText(frame, f'Count: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Frame', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
