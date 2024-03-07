
Pravin Patil <itpravin021@gmail.com>
Wed, Feb 7, 12:30 PM
to SHUBHAM

import cv2
import numpy as np

# Function to calculate distance based on object size in the image
def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Initialize known parameters for distance calculation
KNOWN_DISTANCE = 24.0  # inches
KNOWN_WIDTH = 11.0  # inches
KNOWN_PIXEL_WIDTH = 300.0  # pixels (for the known object width)

# Initialize the camera parameters
FOCAL_LENGTH = (KNOWN_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH

# Create a VideoCapture object to read video frames from a file or camera
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
   
    # Preprocessing
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert the frame from BGR color space to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Thresholding to detect red color
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Contour Detection
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    # Loop through the contours and check for fire-like shapes
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / perimeter ** 2
            if circularity > 0.7:
                # Draw a bounding box around the contour
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate the width of the contour in pixels
                per_width = w

                # Calculate the distance to the object in centimeters
                distance_inches = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, per_width)
                distance_cm = distance_inches * 2.54  # Convert inches to centimeters

                # Display the distance on the frame
                cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()# Personal_Projects
