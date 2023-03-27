import argparse
import platform

import numpy as np
import cv2

import utils.OpenGL
import utils.metal
import utils.image

parser = argparse.ArgumentParser(description="Reconstructs Depth from Focus Stacks")
parser.add_argument("--input", help="Directory containing the focus stack of images")


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

    img = utils.image.LoadImage("resources/images/Seagullz.png")
    imgShape = img.shape
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGaussian = cv2.GaussianBlur(imgGray, (3, 3), 0)
    imgLoG = cv2.Laplacian(imgGaussian, cv2.CV_64F, ksize=3)
    imgLoG = (imgLoG - np.min(imgLoG)) / (np.max(imgLoG) - np.min(imgLoG))
    cv2.imshow("Output", imgLoG)
    cv2.waitKey()

    img = utils.image.FlattenAndScaleImage(img)
    swift_result = compute_focus_metric_function(img)



    cv2.imshow("Output", np.reshape(swift_result, imgShape))
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(swift_result)
    print(swift_result.size)
    utils.metal.cleanup_windows()


if __name__ == "__main__":
    main()
