
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('traffic_sign_model.h5')

# Preprocess image for the model
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

# Define class labels for traffic signs
class_labels = {
    0: 'Speed limit 20',
    1: 'Speed limit 30',
    2: 'Speed limit 50',
    # Add more class labels here
}

# Load the Haar cascade XML file for traffic sign detection
cascade_file = 'traffic_sign_cascade.xml'
cascade = cv2.CascadeClassifier(cascade_file)

# Load the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect traffic signs in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in signs:
        # Extract the region of interest (traffic sign) from the frame
        roi = frame[y:y + h, x:x + w]

        # Preprocess the image
        processed_roi = preprocess_image(roi)

        # Make predictions using the model
        predictions = model.predict(processed_roi)
        predicted_class = np.argmax(predictions[0])
        class_label = class_labels[predicted_class]

        # Draw bounding box and class label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Traffic Sign Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
