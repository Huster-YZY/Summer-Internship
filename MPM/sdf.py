import taichi as ti
import numpy as np
import sys
import time
from tqdm import tqdm
import trimesh
import mesh_to_sdf as md
import mesh2sdf

# ti.init(arch=ti.cpu)

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
pld = None
mesh = None


def load_mesh_fast(filename, n_grid, scale_ratio=1.0):
    global mesh
    global SignedDistanceField
    mesh = trimesh.load(filename, force='mesh')
    size = n_grid * 2
    level = 2 / size
    t0 = time.time()

    mesh.vertices *= 0.5
    mesh.vertices[:, 0] += 0.3

    sdf, _ = mesh2sdf.compute(mesh.vertices,
                              mesh.faces,
                              size,
                              fix=True,
                              level=level,
                              return_mesh=True)
    t1 = time.time()
    print('It takes %.4f seconds to process %s' % (t1 - t0, filename))
    sdf = sdf[-n_grid:, -n_grid:, -n_grid:]

    num_vertices[None] = len(mesh.vertices)
    for i, vertex in enumerate(mesh.vertices):
        vertices[i] = ti.Vector(scale_ratio * vertex)
    for i, face in enumerate(mesh.faces):
        for j in range(3):
            indices[3 * i + j] = face[j]

    return sdf


def gen_point_cloud():
    global pld
    pld = md.get_surface_point_cloud(mesh)


def get_sdf(x):
    global pld
    return pld.get_sdf_in_batches(x)


@ti.kernel
def rotate(ang_x: float, ang_y: float, ang_z: float):
    mat = ti.math.rotation3d(ang_x, ang_y, ang_z)
    for i, j in ti.static(ti.ndrange(3, 3)):
        rotation_matrix[None][i, j] = mat[i, j]

    for i in range(num_vertices[None]):
        vertices[i] = rotation_matrix[None] @ vertices[i]


@ti.kernel
def transform(x: float, y: float, z: float):
    bias = ti.Vector([x, y, z])
    for i in range(num_vertices[None]):
        vertices[i] += bias


# result_dir = "../video"
# video_manager = ti.tools.VideoManager(output_dir=result_dir,
#                                       framerate=60,
#                                       automatic_build=True)


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
        # video_manager.write_frame(window.get_image_buffer_as_numpy())
        window.show()


if __name__ == "__main__":
    filename = './model/cow.obj'
    start_time = time.time()
    load_mesh_fast(filename)
    use_time = time.time() - start_time
    print(f"Load Successfully.\nTime Consumption:{use_time}s")
    main()
