## this is using opencv mode as of now to detect the hotspots

import cv2
import numpy as np


# Initialize the video capture device (adjust the index based on your camera)
cap = cv2.VideoCapture(0)

# Define the threshold for hotspot detection (adjust as needed)
threshold_value = 200

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to detect hotspots
    _, hotspot_mask = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of hotspots
    contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around hotspots and display their temperature
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

        # Calculate the average temperature within the hotspot region
        hotspot_region = gray_frame[y:y+h, x:x+w]
        average_temperature = np.mean(hotspot_region)

        # Display the temperature near the hotspot
        text = f"Temp: {average_temperature:.2f}°C"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow('Thermal Camera', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
