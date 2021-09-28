from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def conv(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    padX = (kW - 1) // 2
    padY = (kH - 1) // 2

    image = cv2.copyMakeBorder(image, padY, padY, padX, padX,
                               cv2.BORDER_REPLICATE)

    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(padY, iH + padY):
        for x in np.arange(padX, iW + padX):
            roi = image[y - padY:y + padY + 1, x - padX:x + padX + 1]
            conv = (roi * kernel).sum()
            output[y - padY, x - padX] = conv

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())


smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
), dtype="int")

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
), dtype="int")

emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss))


image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (name, kernel) in kernelBank:
    print("[INFO] applying {} kernel".format(name))
    convOutput = conv(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    cv2.imshow("Original", gray)
    cv2.imshow("{} - convole".format(name), convOutput)
    cv2.imshow("{} - opencv".format(name), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()