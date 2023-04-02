import argparse
import platform

import numpy as np
import cv2

import utils.OpenGL
import utils.metal
import utils.image

parser = argparse.ArgumentParser(description="Reconstructs Depth from Focus Stacks")
parser.add_argument("--input", help="Directory containing the focus stack of images")
parser.add_argument("--gpu", help="Use GPU for image computation", action=argparse.BooleanOptionalAction)


def main():
    args = parser.parse_args()
    compute_focus_metric_function = None
    match platform.system():
        case 'Darwin':
            print("You are running on MacOS")
            utils.metal.init_compute()
            utils.metal.init_window()
            compute_focus_metric_function = utils.metal.ComputeFocusMetric
        case 'Windows':
            print("You are running on Windows")
            utils.OpenGL.init()
        case 'Linux':
            print("You are running on Linux")
            utils.OpenGL.init()
        case _:
            print("This program does not seem to be running on MacOS, Windows, or Linux. I will now die :(")
            exit(0)


    print("Loading Image Stack...")
    image_stack = utils.image.LoadImageStack(args.input)

    cost_volume = None
    if args.gpu:
        print("Using GPU for Image Computations")
    else:
        cost_volume = utils.image.ComputeCostVolume(image_stack, ksize_L=5, ksize_G=7)

    all_in_focus = utils.image.ComputeAllInFocus(cost_volume, image_stack)
    '''
    img = utils.image.LoadImage("resources/images/Seagullz.png")
    imgShape = img.shape
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGaussian = cv2.GaussianBlur(imgGray, (3, 3), 0)
    imgLoG = cv2.Laplacian(imgGaussian, cv2.CV_64F, ksize=31)
    imgLoG = np.abs(imgLoG)
    imgLoG = (imgLoG - np.min(imgLoG)) / (np.max(imgLoG) - np.min(imgLoG))
    cv2.imshow("Output", imgLoG)
    cv2.waitKey()

    img = utils.image.FlattenAndScaleImage(img)
    swift_result = compute_focus_metric_function(img)
    swift_result = np.reshape(swift_result, imgShape)
    swift_result = (swift_result - np.min(swift_result)) / (np.max(swift_result) - np.min(swift_result))
    '''


    #cv2.imshow("Output", np.reshape(swift_result, imgShape))
    #cv2.waitKey()
    cv2.destroyAllWindows()
    #print(swift_result)
    #print(swift_result.size)
    utils.metal.cleanup_windows()


if __name__ == "__main__":
    main()
