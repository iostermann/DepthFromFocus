import cv2
import os
import numpy as np
import pickle

import utils.metal

image_shape = (0, 0)


def LoadImage(filename: str):
    res = cv2.imread(filename)
    if res is None:
        print("Error Opening Image: ", filename)
    global image_shape
    image_shape = res.shape
    return res


def LoadImageStack(dir_: str):

    # Get list of images in the stack
    stack_file_names = []
    for root, dirs, files in os.walk(dir_):
        for file in files:
            stack_file_names.append(os.path.join(root, file))
    stack_file_names.sort()
    image_stack = {}
    for filename in stack_file_names:
        image_stack[filename] = LoadImage(filename)
    return image_stack


def RegisterImages(stack,
                   method='ECC',
                   order='nearest_first',
                   use_cache=False):
    """
    Registration is Pairwise, so one idea is to move through focal stack and align 1st to 2nd, 2nd to 3rd, etc
    This might cause errors to propagate badly though, so another strategy is to just compare to the image in
    the center of the focal stack??

    order variable specifies if the images go from nearest focus first ro furthest focus first

    Inspired by https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """
    print("Aligning Images with", method, "...")

    if use_cache and os.path.isfile("aligned_stack.pickle"):
        with open('aligned_stack.pickle', 'rb') as handle:
            return pickle.load(handle)
    elif use_cache:
        print("Could not find cache file, aligning images...")

    # Just do pairwise through the stack
    keys = list(stack.keys())

    # Iterate through pairs and align 1-2, 2-3, 3-4, etc
    img_range = None
    if order == 'nearest_first':
        img_range = range(len(keys)-1)
    elif order == 'furthest_first':
        img_range = range(len(keys)-1, 0, -1)

    for i in img_range:
        img1 = stack[keys[i]]
        img2 = stack[keys[i-1]]

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if method == 'ECC':
            homography = np.eye(3, 3, dtype=np.float32)
            iterations = 5000
            eps = 1e-10

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  eps)

            correlation, homography = cv2.findTransformECC(img1_gray,
                                                           img2_gray,
                                                           homography,
                                                           cv2.MOTION_HOMOGRAPHY,
                                                           criteria)
            stack[keys[i-1]] = cv2.warpPerspective(img2,
                                                   homography,
                                                   (image_shape[1], image_shape[0]),
                                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            print("Aligned image", i, "with image", i-1, "with correlation", correlation)

        elif method == 'SIFT':
            # Implementation borrowed heavily from:
            # https://github.com/sangminwoo/Depth_from_Focus/blob/master/Codes/image_alignment.py
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(img2_gray, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img1_gray, None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            matcher = cv2.FlannBasedMatcher(index_params, search_params)

            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []

            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

            # imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

            height, width, channels = img1.shape
            homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            aligned_img = cv2.warpPerspective(img2, homography, (width, height))
            stack[keys[i - 1]] = aligned_img
            print("Aligned image", i, "with image", i-1, "with", len(good_matches), "SIFT features")

    # for filename, img in stack.items():
    #     cv2.imshow(filename, img)
    #     cv2.waitKey(0)

    print("Saving aligned images for caching purposes")
    with open('aligned_stack.pickle', 'wb') as handle:
        pickle.dump(stack, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stack


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
        # img_result = cv2.GaussianBlur(img_result, (15, 15), 0)
        cost_volume[i] = img_result
        i += 1
    return cost_volume


def ComputeAllInFocus(cost_volume, stack):
    print("Assembling All-In-Focus Image...")
    all_in_focus = np.zeros((image_shape[0], image_shape[1], 3), dtype='uint8')
    max_focus = np.argmax(cost_volume, axis=0)

    # Blur it a bit so things stay locally correct
    max_focus = cv2.bilateralFilter(max_focus.astype("float32"), 9, 75, 75).astype("uint8")

    # Convert Stack to something better indexable
    stack_volume = np.zeros((len(stack), image_shape[0], image_shape[1], 3), dtype='uint8')
    i = 0
    for img in stack.values():
        stack_volume[i] = img
        i += 1

    all_in_focus = utils.metal.ComputeAllInFocus(stack_volume, max_focus)
    all_in_focus = np.reshape(all_in_focus, (image_shape[0], image_shape[1], 3))

    # This MUST be vectorizable??
    #for x in range(image_shape[0]):
    #    for y in range(image_shape[1]):
    #        all_in_focus[x][y] = stack_volume[max_focus[x][y]][x][y]


    # Scale, convert, then flip range to pretty print the depth
    max_focus = max_focus.astype('float')
    max_focus = (max_focus - np.min(max_focus)) / (np.max(max_focus) - np.min(max_focus))
    max_focus *= 255
    max_focus = 255 - max_focus
    max_focus_pretty = cv2.applyColorMap(max_focus.astype('uint8'), cv2.COLORMAP_VIRIDIS)
    cv2.imshow("Output", max_focus_pretty)
    cv2.waitKey(0)
    cv2.imshow("Output", all_in_focus)
    cv2.waitKey(0)
    return all_in_focus
