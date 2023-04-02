import ctypes
import numpy as np
import cv2

swift_fun = ctypes.CDLL


def init_window():
    print("Initializing Windowing System")
    cv2.startWindowThread()
    cv2.namedWindow("Output")


def cleanup_windows():
    cv2.destroyAllWindows()


def init_compute():
    print("Initializing Metal backend for Compute")
    global swift_fun
    swift_fun = ctypes.CDLL("./PyMetalBridge/.build/release/libPyMetalBridge.dylib")
    swift_fun.swift_focus_metric_flat_on_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    return swift_fun


def ComputeFocusMetric(input_image: np.ndarray):
    input_ptr = input_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_mutable_ptr = (ctypes.c_float * len(input_image))()
    swift_fun.swift_focus_metric_flat_on_gpu(input_ptr, output_mutable_ptr, len(input_image))
    return np.array(output_mutable_ptr)
