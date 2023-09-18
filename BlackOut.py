import imutils
import cv2
import numpy as np

image = cv2.imread("Will.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

w = 1280
h = 720

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('Facial.mp4', fourcc, 20.0, (640, 480))
out1 = cv2.VideoWriter('Facial.mp4', 0x7634706d, 30.0, (640, 480))


# frame counter
video_frame = 0
print("loading model..")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
print("Done")
while video_frame < 100:
    ret, frame = cap.read()
    print(frame.shape)



    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            bbox = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = bbox.astype("int")
            text = "{:.2f}%".format(confidence*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (156, 208, 30), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (174, 0, 255))


    cv2.imshow("Frame", frame)
    cv2.waitKey(33)
    video_frame += 1
    out1.write(frame)

cap.release()
cv2.destroyAllWindows()