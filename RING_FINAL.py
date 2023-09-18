import numpy as np
import imutils
import cv2

def mask_image(img, pts):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")  # intialize the mask
    cv2.fillConvexPoly(mask, pts, 255)
    mask = cv2.bitwise_and(img, img, mask=mask)
    return mask

def mouse_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(len(params)):
            if np.sqrt((params[i][0] - x)**2 + (params[i][1] - y)**2) < 10:
                params[0] = i
                break
        else:
            params[0] = -1
    elif event == cv2.EVENT_MOUSEMOVE and params[0] != -1:
        params[params[0]] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        params[0] = -1

cv2.namedWindow('image')

pts = [(10, 10), (10, 360), (10, 710), (540, 710), (1270, 710), (1270, 360), (1270, 10), (540, 10)]

cv2.setMouseCallback('image', mouse_event, [0])

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('RING.mp4', fourcc, 20.0, (640, 480))
timer = False
First = True
while True:
    # grab the current frame
    ret, frame = camera.read()
    if not First:
        mask = mask_image(frame, np.array(pts))
        maskPrev = mask_image(prevFrame, np.array(prevPts))
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        greyPrev = cv2.cvtColor(maskPrev, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)  # Gaussian blur of 5x5
        blurPrev = cv2.GaussianBlur(greyPrev, (5, 5), 0)  # Gaussian blur of 5x5
        # calculate the absolute difference between the frames
        diff = cv2.absdiff(blur, blurPrev)
        # apply thresholding to the difference image
        thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
        threshold_value = 50
        if (cv2.countNonZero(thresh) > threshold_value) and not timer:
            print('start recording')
            timer = True
            frames = 300;
        if timer:
            frames = frames -1
            out.write(frame)
            # record video and upload
            # notify the user
        if frames == 0:
            timer = False

    First = False
    prevFrame = frame.copy()
    prevPts = pts.copy()
    img = np.zeros_like(frame)
    cv2.fillConvexPoly(img, np.array(pts), (0, 0, 255))

    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 10, (0, 255, 0), -1)
        cv2.putText(img, str(i), (pt[0] - 5, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    res = cv2.addWeighted(frame, 0.5, img, 0.5, 0)

    cv2.imshow('image', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
out.release()
cv2.destroyAllWindows()