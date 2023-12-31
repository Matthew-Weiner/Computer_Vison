# ENME 489Y: Remote Sensing
# Code uses laptop webcam to track objects
# Import the necessary packages
import numpy as np
import imutils
import cv2

# Define the lower and upper boundaries of the
# green light (circle) in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 70, 6)
greenUpper = (75, 255, 255)
yellowLower = (20, 100, 100)
yellowUpper = (40, 255, 255)
redLower = (0, 100, 100)
redUpper = (10, 255, 255)
y = False
g = False
r = True
if g:
    Low = greenLower
    Up = greenUpper
    Color = 'green'
if y:
    Low = yellowLower
    Up = yellowUpper
    Color = 'Yellow'
if r:
    Low = redLower
    Up = redUpper
    Color = "red"

tracked_points = []

# initialize the webcam
camera = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
camera.set(3, 640)
camera.set(4, 480)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('Red.mp4', fourcc, 20.0, (640, 480))


while True:
    # grab the current frame
    ret, frame = camera.read()

    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, Low, Up)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       Low, 2)
            cv2.circle(frame, center, 5, Up, -1)

            # add the center to the tracked points list
            tracked_points.append(center)

            # draw a circle at the center of the green color
            cv2.circle(frame, center, int(radius), Low, -1)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # define the font
            cv2.putText(frame, f'Tracking {Color}', (0, 20), font, 2, (0, 0, 0), 2)
            # draw the path of the circle
            for i in range(1, len(tracked_points)):
                cv2.line(frame, tracked_points[i-1], tracked_points[i], Low, thickness=2)

            # # update the start time
            # if center is not None:
            #     start_time = time.time()

    cv2.imshow("Frame", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
out.release()
cv2.destroyAllWindows()
