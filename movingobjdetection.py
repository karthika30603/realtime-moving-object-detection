import cv2
import imutils
import time

# Open the camera
cam = cv2.VideoCapture(0)
time.sleep(1)

# Initialize variables
first_screen = None
area = 500

while True:
    # Read the current frame from the camera
    _, frame = cam.read()

    # Resize the frame
    frame = imutils.resize(frame, width=500)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to the frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # If this is the first frame, save it
    if first_screen is None:
        first_screen = blurred_frame
        continue

    # Compute the absolute difference between the current frame and the first frame
    frame_diff = cv2.absdiff(first_screen, blurred_frame)

    # Threshold the difference
    thresh_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image
    dilated_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours in the thresholded image
    contours = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loop over the contours
    for contour in contours:
        if cv2.contourArea(contour) < area:
            continue

        # Get the bounding box coordinates
        (x, y, w, h) = cv2.boundingRect(contour)

        # Draw a rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display a text indicating a moving object
        cv2.putText(frame, "Moving Object Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Camera Feed", frame)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()