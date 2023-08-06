import taichi as ti
import numpy as np
import tinyobjloader
import sys
import time
from tqdm import tqdm

ti.init(arch=ti.cpu)

MAX_VERTEX_NUM = 25000
FACE_NUM = 50000
MAX_FACE_NUM = 3 * FACE_NUM

window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

vertices = ti.Vector.field(3, dtype=float, shape=MAX_VERTEX_NUM)
indices = ti.field(dtype=int, shape=MAX_FACE_NUM)
num_vertices = ti.field(dtype=int, shape=())
rotation_matrix = ti.Matrix.field(3, 3, dtype=float, shape=())
PI = 3.1415926


def load_mesh(filename, scale_ratio=5.0):
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
    num_vertices[None] = vertex_length

    for i in tqdm(range(vertex_length)):
        vertices[i] = scale_ratio * ti.Vector([
            attrib.vertices[i * 3], attrib.vertices[i * 3 + 1],
            attrib.vertices[i * 3 + 2]
        ])

    #Load Mesh Indices
    shapes = reader.GetShapes()
    offset = 0
    for shape in shapes:
        assert len(shape.mesh.indices) % 3 == 0
        faces_length = len(shape.mesh.indices) // 3

        #speed bottleneck
        for i in tqdm(range(faces_length)):
            indices[offset + i * 3] = shape.mesh.indices[i * 3].vertex_index
            indices[offset + i * 3 + 1] = \
                shape.mesh.indices[i * 3 +1].vertex_index
            indices[offset + i * 3 + 2] = \
                shape.mesh.indices[i * 3 +2].vertex_index

        offset += faces_length * 3


@ti.kernel
def rotate(ang_x: float, ang_y: float, ang_z: float):
    mat = ti.math.rotation3d(ang_x, ang_y, ang_z)
    for i, j in ti.static(ti.ndrange(3, 3)):
        rotation_matrix[None][i, j] = mat[i, j]

    for i in range(num_vertices[None]):
        vertices[i] = rotation_matrix[None] @ vertices[i]


result_dir = "../video"
video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                      framerate=60,
                                      automatic_build=True)


def main():
    while window.running:
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)

        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(vertices, indices, two_sided=True)
        angle = 0.05
        rotate(0, 0, angle)
        canvas.scene(scene)
        video_manager.write_frame(window.get_image_buffer_as_numpy())
        window.show()


if __name__ == "__main__":
    filename = './model/bunny.obj'
    start_time = time.time()
    load_mesh(filename)
    use_time = time.time() - start_time
    print(f"Load Successfully.\nTime Consumption:{use_time}s")
    main()
