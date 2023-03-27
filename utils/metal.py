import ctypes


def init_window():
    print("Initializing Windowing System")


def init_compute():
    print("Initializing Metal backend for Compute")
    swift_fun = ctypes.CDLL("./PyMetalBridge/.build/release/libPyMetalBridge.dylib")
