import argparse
import platform
import utils.OpenGL
import utils.metal

parser = argparse.ArgumentParser(description="Reconstructs Depth from Focus Stacks")
parser.add_argument("--input", help="Directory containing the focus stack of images")


def main():
    args = parser.parse_args()

    match platform.system():
        case 'Darwin':
            print("You are running on MacOS")
            utils.metal.init_compute()
            utils.metal.init_window()
        case 'Windows':
            print("You are running on Windows")
            utils.OpenGL.init()
        case 'Linux':
            print("You are running on Linux")
            utils.OpenGL.init()
        case _:
            print("This program does not seem to be running on MacOS, Windows, or Linux. I will now die :(")
            exit(0)


if __name__ == "__main__":
    main()
