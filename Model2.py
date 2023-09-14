import cv2
import numpy as np

# Load the image
image_path = "transformer_image.jpg"  # Replace with the path to your transformer image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Failed to load the image.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the detected contours on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    # Display the original image with contours
    cv2.imshow("Transformer Image with Contours", result_image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
