import imutils
import cv2
import numpy as np

w = 1280
h = 720

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('Facial.mp4', fourcc, 20.0, (640, 480))
out1 = cv2.VideoWriter('Facial.mp4', 0x7634706d, 30.0, (w, h))


# frame counter
video_frame = 0
print("loading model..")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
print("Done")
while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            bbox = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = bbox.astype("int")
            text = "{:.2f}%".format(confidence*100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # Start determining the center
            xRange = endX - startX
            yRange = endY - startY
            xPos = 640 - ((startX + endX) //2 )
            yPos = 360 - ((startY + endY) //2 )
            if xPos > 0:
                left = 640 - abs(xPos)
                right = 640 + abs(xPos)
            else:
                left = 640 + abs(xPos)
                right = 640 - abs(xPos)
            if yPos > 0:
                top = 360 - abs(yPos)
                bottom = 360 + abs(yPos)
            else:
                top = 360 + abs(yPos)
                bottom = 360 - abs(yPos)



            frame = np.pad(frame, ((bottom, top), (right, left), (0, 0)), 'constant')
            startX = 1280 - (xRange//2)
            startY = 720 - (yRange//2)
            endX = 1280 + (xRange//2)
            endY = 720  + (yRange//2)
            y = endY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (174, 0, 255))


    cv2.imshow("Frame", frame)
    cv2.waitKey(33)
    video_frame += 1
    out1.write(frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

out1.release()
cap.release()
cv2.destroyAllWindows()