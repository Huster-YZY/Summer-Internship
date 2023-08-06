import taichi as ti
import numpy as np
import tinyobjloader
import sys

ti.init(arch=ti.metal)

VERTEX_NUM = 2000
FACE_NUM = 5000
MAX_VERTEX_NUM = 3 * VERTEX_NUM
MAX_FACE_NUM = 3 * FACE_NUM

window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

vertices = ti.Vector.field(3, dtype=float, shape=MAX_VERTEX_NUM)
indices = ti.field(dtype=int, shape=MAX_FACE_NUM)

ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]
ball_radius = 0.3


def load_mesh(filename):
    reader = tinyobjloader.ObjReader()
    ret = reader.ParseFromFile(filename)
    if ret == False:
        print("Warn:", reader.Warning())
        print("Err:", reader.Error())
        print("Failed to load : ", filename)

        sys.exit(-1)
    # Load Vertices
    attrib = reader.GetAttrib()
    assert len(attrib.vertices) % 3 == 0
    vertex_length = len(attrib.vertices) // 3

    for i in range(vertex_length):
        vertices[i + 1] = 100 * ti.Vector([
            attrib.vertices[i * 3], attrib.vertices[i * 3 + 1],
            attrib.vertices[i * 3 + 2]
        ])

    #Load Mesh Indices
    shape = reader.GetShapes()[0]
    assert len(shape.mesh.indices) % 3 == 0
    faces_length = len(shape.mesh.indices) // 3

    #!!! ERROR
    for i in range(faces_length):
        indices[i * 3] = shape.mesh.indices[i * 3].vertex_index
        indices[i * 3 + 1] = shape.mesh.indices[i * 3 + 1].vertex_index
        indices[i * 3 + 2] = shape.mesh.indices[i * 3 + 2].vertex_index


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
    filename = './model/block.obj'
    load_mesh(filename)
    main()
