import ctypes
import numpy as np
import cv2

from time import time

swift_fun = ctypes.CDLL


def timeit(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


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
    swift_fun.swift_all_in_focus_on_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),

    ]
    return swift_fun


def ComputeFocusMetric(input_image: np.ndarray):
    input_ptr = input_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_mutable_ptr = (ctypes.c_float * len(input_image))()
    swift_fun.swift_focus_metric_flat_on_gpu(input_ptr, output_mutable_ptr, len(input_image))
    return np.array(output_mutable_ptr)


def ComputeAllInFocus(imageVolume: np.ndarray, focusIndices: np.ndarray):
    imageVolume_flat = imageVolume.ravel().astype("float32")
    imageVolume_flat_ptr = imageVolume_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    focusIndices_flat = focusIndices.ravel().astype("int")
    focusIndices_flat_ptr = focusIndices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    shape = imageVolume.shape
    L_ptr = ctypes.pointer(ctypes.c_int(shape[2]))
    W_ptr = ctypes.pointer(ctypes.c_int(shape[1]))
    D_ptr = ctypes.pointer(ctypes.c_int(shape[0]))

    output_mutable_ptr = (ctypes.c_float * (len(focusIndices_flat) * 3))()

    swift_fun.swift_all_in_focus_on_gpu(imageVolume_flat_ptr,
                                        focusIndices_flat_ptr,
                                        output_mutable_ptr,
                                        L_ptr,
                                        W_ptr,
                                        D_ptr)

    return np.array(output_mutable_ptr, dtype="uint8")

