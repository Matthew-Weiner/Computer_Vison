import imutils
import cv2
import numpy as np
import random

# question one only needs one input, the string for the name of the image
def q_1(name):
    # part 1
    image = cv2.imread(name)  # read in the image

    # part 2
    image = imutils.resize(image, width=400)   # make the image  400x400
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL    # define the font
    cv2.putText(image, 'Matthew Weiner', (0, 20), font, 1, (255, 255, 255), 1)
    # put my name as text on the loaded image in white
    cv2.imshow("Part 2", image)   # Show the new image
    cv2.imwrite("Part_2.jpeg", image)
    # Part 3
    # every 4th column and 4th row
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if i % 4 == 0 and j % 4 == 0:   # if both dimensions are divisble by 4, proceed
                (b, g, r) = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # generate random bgr
                image[i, j] = (b, g, r)  # asign bgr value to image dimension
    cv2.imshow("Part 3", image)
    cv2.imwrite("Part_3.jpeg", image)

    # Part 4
    def median_blur_bgr(image, size):
        # median blur function, which has two input. Image as a string and dimension of square kernel
        if size % 2 == 0:    # in most convolution algorithims, the kernel dimensions must be odd.
            print("Size parameter must be odd.")
            return
        pad_size = (size - 1) // 2     # pad size is the amount of the image which would be cut off.
        blurred = np.zeros(image.shape, dtype=np.uint8)  # assign an empty canvas which the median image will be painted
        imagepad = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
        # This line pads the input image array with a border of whatever the outer number was (using the edge pixels)
        # so that the output blurred image has the same size as the input image.
        for c in range(image.shape[2]):  # Three color dimensions
            for i in range(image.shape[1]):  # length
                for j in range(image.shape[0]):  # width
                    patch = imagepad[i:i + size, j:j + size, c]  # look at the "patch" which the kernel covers
                    blurred[i, j, c] = np.median(patch)  # find the median of the value/assign it to the center location
        cv2.imshow(f"Part 4 with Color: {size}x{size}", blurred)
        cv2.imwrite(f"Part_4_with Color:_{size}x{size}.jpeg", blurred)
        return
    median_blur_bgr(image, 3)
    median_blur_bgr(image, 4) # This won't work, given my constraint in the function that size must be odd
    median_blur_bgr(image, 5)
    # The greater the blur size the more quality of the original image is lost.

    cv2.waitKey(0)
    return


q_1('yoda.jpeg')





