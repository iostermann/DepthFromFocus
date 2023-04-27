import argparse
import platform

import cv2

import utils.image
import utils.viewer

parser = argparse.ArgumentParser(description="Reconstructs Depth from Focus Stacks")
parser.add_argument("--input", help="Directory containing the focus stack of images")
parser.add_argument("--gpu", help="Use GPU for image computation", action=argparse.BooleanOptionalAction)
parser.add_argument("--cache", help="Look for a cache of the registered images", action=argparse.BooleanOptionalAction)


def main():
    args = parser.parse_args()
    compute_focus_metric_function = None
    platformType = platform.system()
    if platformType == 'Darwin':
        print("You are running on MacOS")
        import utils.metal

        utils.metal.init_compute()
        utils.metal.init_window()
        compute_focus_metric_function = utils.metal.ComputeFocusMetric
    elif platformType == 'Windows':
        import utils.OpenGL
        print("You are running on Windows")
        utils.OpenGL.init()
    elif platformType == 'Linux':
        import utils.OpenGL
        print("You are running on Linux")
        utils.OpenGL.init()
    else:
        print("This program does not seem to be running on MacOS, Windows, or Linux. I will now die :(")
        exit(0)

    print("Loading Image Stack...")
    image_stack = utils.image.LoadAndRegisterImages(args.input, method='SIFT', order='furthest_first', use_cache=args.cache)

    cost_volume = None
    cost_volume = utils.image.ComputeCostVolume(image_stack, ksize_L=5, ksize_G=9)

    all_in_focus, depth = utils.image.ComputeAllInFocus(cost_volume, image_stack, use_gpu=args.gpu)

    depth = 255 - depth
    cv2.imwrite("depth.png", depth)
    cv2.imwrite("all-in-focus.png", all_in_focus)

    utils.viewer.show_depth_image()

    cv2.destroyAllWindows()
    utils.metal.cleanup_windows()


if __name__ == "__main__":
    main()
