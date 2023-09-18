import numpy as np
import imutils
import cv2

# read video from file
vid1 = cv2.VideoCapture('test_video_01.mp4')
# read video from file
vid2 = cv2.VideoCapture('test_video_02.mp4')

for vid in [vid1, vid2]:     # this is a check to make sure the videos loaded properly (25 fps?? get a better camera)
    video_fps = vid.get(cv2.CAP_PROP_FPS),
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    print(f"Frame Per second: {video_fps} \nTotal Frames: {total_frames} \n Height: {height} \nWidth: {width}")
# This tells us the videos are the same size, so I can use the exact same mask for both


def mask_image(img):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")  # intialize the mask
    pts = np.array([[40, img.shape[0]-65], [40, img.shape[0]-75], [int(img.shape[1]/2)-30, 30], [int(img.shape[1]/2)+30, 30], [img.shape[1]-40, img.shape[0]-75], [img.shape[1]-40, img.shape[0]-65]])
    # I changed this to only include the lane you are in.
    cv2.fillConvexPoly(mask, pts, 255)
    mask = cv2.bitwise_and(img, img, mask=mask)
    return mask


def find_lines(img):
    lines = cv2.HoughLines(img, 1, np.pi/180, 40 ) #(image, radius, radians, voting threshold)
    # Changing the Voting threshold to a lower number results in a noiser solution, because more lines are drawn
    return lines

def calc(img, rho, theta):
    # this function was created to change coordinates from polar to cartesian
    a = np.cos(theta);
    b = np.sin(theta)
    x0 = a * rho;
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b));
    y1 = int(y0 + 1000 * a + img.shape[0] / 2)
    x2 = int(x0 - 1000 * (-b));
    y2 = int(y0 - 1000 * a + img.shape[0] / 2)  # equations from class
    # (x1, y1) and (x2, y2) are the points to draw the lines on the entire screen. However, we only want the lines on
    #part of the screen. So we create ymin and ymax as the minimum and maximmum height of the lines so they are always
    # the same.
    ymax = int(2 * (img.shape[0] / 3))
    ymin = int(img.shape[0])
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    x1 = int((ymax - intercept) / slope)
    x2 = int((ymin - intercept) / slope)
    return x1, x2, ymax, ymin


def plot_hough_lines(img, rho, theta):
    if rho > 0:     # set the colors. If rho>0 then it is a left lane (red)
        color = (0, 0, 255)
    else:   # right lane blue
        color = (255, 0, 0)
    x1, x2, ymax, ymin = calc(img, rho, theta)  # calculate the line pints
    cv2.line(img, (x1, ymax), (x2, ymin), color, 8)  # draw a line using the specified points
    return img


def get_average_line(lines, previous_lines):
    rL = 0; rR = 0; tL = 0; tR = 0;   # initialize variables which will serve as the mean of the hough lines
    i = 0; j = 0;   # this will count how many left or right lines there are
    if lines is not None:
        for k in range(0, len(lines)):
            for rho, theta in lines[k]:
                if 2.3 > theta > 2.1:
                    # since the lines are usually -135 or 135, only include these lines so there are no outliers
                    i = i+1
                    rL = rL + rho  # add all the values in order to find the mean
                    tL = tL + theta
                    prev = lines
                elif 1.1 > theta > .85:
                    # same here, only include lines with a resonable angle
                    j = j+1
                    rR = rR + rho    # add all the values in order to find the mean
                    tR = tR + theta
                    prev = lines
    if i == 0:
        # This is incase the frame says there is no left or right lane
        # instead of showing no lane, I am using the hough lines for the last frame which had those lanes
        for k in range(0, len(previous_lines)):
            for rho, theta in previous_lines[k]:
                if 2.3 > theta > 2.1:
                    i = i + 1
                    rL = rL + rho
                    tL = tL + theta
                    prev = previous_lines

    if j == 0:
        # This is incase the frame says there is no left or right lane
        # instead of showing no lane, I am using the hough lines for the last frame which had those lanes
        for k in range(0, len(previous_lines)):
            for rho, theta in previous_lines[k]:
                if 1.1 > theta > .85:
                    j = j + 1
                    rR = rR + rho
                    tR = tR + theta
                    prev = previous_lines
    right_rho = rR/j
    left_rho = rL/i
    right_theta = tR/j
    left_theta = tL/i
    return right_rho, left_rho, right_theta, left_theta, prev

def fill(img, right_rho, left_rho, right_theta, left_theta):
    x7, x8, ymax, ymin = calc(img, right_rho, right_theta)
    x3, x4, ymax, ymin = calc(img, left_rho, left_theta)
    pts = np.array([[x7, ymax], [x3, ymax], [x4, ymin], [x8, ymin]])
    cv2.fillConvexPoly(img, pts, (0, 255, 0))   # this function creates the green mask on top of the frames.
    return img


def q1(vid):
    previous_lines = [[[154.0, .9777], [-128, 2.20]]]  # this is the initial line, incase there are no left or right
    # lane lines detected in the first frame
    while True:
        ret, frame = vid.read()
        if not ret:
            break  # break if no next frame
        snip = frame[int(frame.shape[0]/2):int(frame.shape[0]), 0:int(frame.shape[1])]
        # snip the image to only take the bottom half to reduce processing time
        mask = mask_image(snip)  # create the mask on the snip
        HSV = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)   # Convert image to HSV
        whiteLower = np.array([0, 0, 50]) # define threshold for white in HSV
        whiteUpper = np.array([75, 75, 255])
        # whiteLower = np.array([190])
        # whiteUpper = np.array([255])    # If I used grey scale
        white = cv2.inRange(HSV, whiteLower, whiteUpper)
        blur = cv2.GaussianBlur(white, (5, 5), 0)  #Gaussian blur of 5x5
        # Canny Edge Detection
        edges = cv2.Canny(image=blur, threshold1=100, threshold2=200)  # Canny Edge Detection
        lines = find_lines(edges)  # find hough lines based on canny edge detection
        right_rho, left_rho, right_theta, left_theta, previous_lines = get_average_line(lines, previous_lines)
        if lines is not None:
            frame2 = frame.copy()  # creat a copy of the frame which will be used to create a final image w opacity
            plot_hough_lines(frame, right_rho, right_theta)
            plot_hough_lines(frame, left_rho, left_theta)    # plot hough lines on the frame
            fill(frame, right_rho, left_rho, right_theta, left_theta)  # fill in green path
            res = cv2.addWeighted(frame2, .5, frame, 0.5, 0)  # Overlay the plotted lines and fill on orginal frame 50%
            cv2.imshow("Hough Lines", res)

        cv2.imshow('Frame', edges)  # show frame

        if cv2.waitKey(25) & 0xFF == ord('q'):  # on press of q break
            break
    # release and destroy windows
    vid.release()
    # out.release()
    cv2.destroyAllWindows()
    return


q1(vid1)  # This is used to play the function for 'vid1' or 'vid2'
