import cv2 as cv

# Create a VideoCapture object, 0 specifies the first camera
cap = cv.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is not read correctly, ret is False
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('Live Camera Feed', frame)

    # Break the loop when the 'q' key is pressed
    # cv.waitKey(1) waits for 1 millisecond between frames
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture and destroy windows
cap.release()
cv.destroyAllWindows()