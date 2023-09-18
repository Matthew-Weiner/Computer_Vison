
import imutils
import cv2
import numpy as np
import qrcode
import os



# Define the class of people with different attributes and abilities
class Person:
    def __init__(self, id, privileges):
        self.id = id
        self.privileges = privileges

# Create QR codes for each person
person1 = Person("ID001", ["door1"])
person2 = Person("ID002", ["door2"])
person3 = Person("ID003", ["door1", "door2"])
person4 = Person("ID004", [])

# Only Need to run this once, can comment out after it is created
# code = qrcode.make('ID001')
# code.save('person1.png')
# code = qrcode.make('ID002')
# code.save('person2.png')
# code = qrcode.make('ID003')
# code.save('person3.png')
# code = qrcode.make('ID004')
# code.save('person4.png')


# Set which door you are trying to access
door_to_access = "door1"

# Start the camera
cap = cv2.VideoCapture(0)

detector = cv2.QRCodeDetector()

# Initialize error counter and person tracking variables
error_count = 0
last_person_id = None
same_person_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and decode a single barcode
    data, bbox, _ = detector.detectAndDecode(frame)
    if bbox is not None:
        # Draw the bounding box around the barcode
        n_bboxes = len(bbox)
        for i in range(n_bboxes):
            # cv2.rectangle(frame, tuple(bbox[i][0]), tuple(bbox[(i + 1) % len(bbox)][0]), color=(0, 0, 255), thickness=4)
            frame = cv2.polylines(frame, bbox.astype(int), True, (0, 255, 0), 3)
            cv2.rectangle(frame, tuple(map(int, bbox[i][0])), tuple(map(int, bbox[(i + 1) % len(bbox)][0])),
                          color=(0, 0, 255), thickness=4)
            cv2.putText(frame, data, (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2 )
        print("Decoded data: ", data)

        # Determine which person it is
        if data == person1.id:
            current_person = person1
        elif data == person2.id:
            current_person = person2
        elif data == person3.id:
            current_person = person3
        elif data == person4.id:
            current_person = person4
        else:
            current_person = None

        if current_person is not None:
            # Check their privileges
            if door_to_access in current_person.privileges:
                # Open door
                print("Access granted. Opening door...")
                # TODO: Add code to actually open the door
                same_person_count = 0
            else:
                # Give error that you don't have access
                print("Access denied. You don't have access to this door.")
                error_count += 1

                # If the same person tries three times, alert someone
                if current_person.id == last_person_id:
                    same_person_count += 1
                else:
                    same_person_count = 1
                    last_person_id = current_person.id

                if same_person_count == 3:
                    print("Same person tried three times. Alerting security...")
                    for i in range(3):
                        voice = 'Bubbles'  # replace with the name of the voice you want to use
                        text = 'Warning, Warning, Warning      '
                        os.system(f'say -v {voice} "{text}"')
            # Reset error count if a different person tries
            if current_person.id != last_person_id:
                error_count = 0

        # Show the image
        cv2.imshow("Image", frame)
        # Wait for key press to exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
