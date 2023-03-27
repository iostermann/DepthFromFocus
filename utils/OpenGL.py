import moderngl as mgl
import moderngl_window as mglw
from moderngl_window import geometry
from pathlib import Path


class Window(mglw.WindowConfig):
    gl_version = (4, 1)
    title = "Depth From Focus"
    resource_dir = (Path(__file__) / '../../resources').resolve()
    # window_size = (1920, 1080)
    aspect_ratio = 1.0
    # resizable = False
    # samples = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_shader = self.load_compute_shader("shaders/DFF_cs.glsl")
        self.compute_shader['destTex'] = 0
        self.texture_program = self.load_program("shaders/DFF_texture.glsl")
        self.quad_fs = geometry.quad_fs()
        self.texture = self.ctx.texture((256, 256), 4)
        self.texture.filter = mgl.NEAREST, mgl.NEAREST

    def render(self, time: float, frametime: float):
        self.ctx.clear(0.3, 0.3, 0.3)

        w, h = self.texture.size
        gw, gh = 16, 16
        nx, ny, nz = int(w / gw), int(h / gh), 1

        try:
            self.compute_shader['time'] = time
        except Exception:
            pass
        # Automatically binds as a GL_R32F / r32f (read from the texture)
        self.texture.bind_to_image(0, read=False, write=True)
        self.compute_shader.run(nx, ny, nz)

        # Render texture
        self.texture.use(location=0)
        self.quad_fs.render(self.texture_program)

    def resize(self, width: int, height: int):
        print("Window was resized. buffer size is {} x {}".format(width, height))

    def mouse_position_event(self, x, y, dx, dy):
        # print("Mouse position:", x, y)
        pass

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))

    def mouse_release_event(self, x: int, y: int, button: int):
        print("Mouse button {} released at {}, {}".format(button, x, y))

    def key_event(self, key, action, modifiers):
        print(key, action, modifiers)


def init():
    print("Initializing OpenGL")
    mglw.run_window_config(Window)
