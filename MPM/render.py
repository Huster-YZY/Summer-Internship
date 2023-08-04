import taichi as ti
import numpy as np

ti.init(arch=ti.metal)

window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
vertices = ti.Vector.field(3, dtype=float, shape=(3))
indices = ti.field(dtype=int, shape=(3))

a = np.array([(0.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 1.0, 0.0)])
b = np.array([[0, 1, 2]])

ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]
ball_radius = 0.3


def load_mesh(a: ti.types.ndarray(), b: ti.types.ndarray()):
    for i in range(3):
        vertices[i] = ti.Vector(a[i])
        indices[i] = b[0, i]


def main():

    while window.running:
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)

        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(vertices, indices, two_sided=True)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    load_mesh(a, b)
    main()
