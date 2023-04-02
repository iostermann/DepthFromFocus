import cv2
import os
import numpy as np

image_shape = (0, 0)

def LoadImage(filename: str):
    res = cv2.imread(filename)
    if res is None:
        print("Error Opening Image: ", filename)
    global image_shape
    image_shape = res.shape
    return res


def LoadImageStack(dir: str):

    # Get list of images in the stack
    stack_file_names = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            stack_file_names.append(os.path.join(root, file))
    stack_file_names.sort()
    image_stack = {}
    for filename in stack_file_names:
        image_stack[filename] = LoadImage(filename)
    return image_stack


def LoadImageGrayscale(filename: str):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def FlattenAndScaleImage(img: np.ndarray):
    img = img.ravel().astype("float32")
    img /= 255.0
    return img


def ComputeCostVolume(stack, ksize_L=3, ksize_G=3):

    print("Computing Focus Metric Volume...")
    cost_volume = np.zeros((len(stack), image_shape[0], image_shape[1]))
    i = 0
    for filename, img in stack.items():
        # Slightly Blur Image
        img_blur = cv2.GaussianBlur(img, (ksize_G, ksize_G), 0)

        # Convert Image to greyscale
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)


        # Compute Laplacian
        img_dst = cv2.Laplacian(img_gray, ddepth=cv2.CV_16S, ksize=ksize_L)

        # Scale back to uint8
        img_result = cv2.convertScaleAbs(img_dst)
        cost_volume[i] = img_result
        i += 1
    return cost_volume


def ComputeAllInFocus(cost_volume, stack):
    print("Assembling All-In-Focus Image...")
    all_in_focus = np.zeros((image_shape[0], image_shape[1], 3), dtype='uint8')
    max_focus = np.argmax(cost_volume, axis=0)

    # Convert Stack to something better indexable
    stack_volume = np.zeros((len(stack), image_shape[0], image_shape[1], 3))
    i = 0
    for img in stack.values():
        stack_volume[i] = img
        i += 1

    # This MUST be vectorizable
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            all_in_focus[x][y] = stack_volume[max_focus[x][y]][x][y]
    cv2.imshow("Output", all_in_focus)
    cv2.waitKey(0)
