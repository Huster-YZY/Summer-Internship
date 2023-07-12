import taichi as ti

ti.init(arch=ti.cpu)

# the scale of the simulation
n = 128
use_implicit = False
iter_number = 3
quad_size = 1.0 / n

# dt = 3.125e-4
dt = 4e-4 / n

# substeps=53,the number of iteration if we want to achieve 60fps
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
I = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
spring_Y = 3
dashpot_damping = 1
drag_damping = 1

# balls' model
ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(3, ))
ball_center[0] = [0, 0, 0]
ball_center[1] = [0.4, 0, 0]
ball_center[2] = [-0.4, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
num_node = int(n * n)
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

#Implicit method
F = ti.Vector.field(3, dtype=ti.f32, shape=num_node)
J = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(num_node, num_node))
A = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(num_node, num_node))
M = ti.field(dtype=ti.f32, shape=(num_node, num_node))
new_v = ti.Vector.field(3, dtype=ti.f32, shape=num_node)
b = ti.Vector.field(3, dtype=ti.f32, shape=num_node)

# configuration
bending_springs = False


@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = i * (n - 1) + j
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)

        indices[quad_id * 6 + 3] = (i + 1) * n + (j + 1)
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)


initialize_mesh_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))
else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))


@ti.kernel
def collision_detection():
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        for j in range(3):
            offset_to_center = x[i] - ball_center[j]
            if offset_to_center.norm() <= ball_radius:
                normal = offset_to_center.normalized()
                v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += v[i] * dt


@ti.func
def m2v(x):
    return x[0] * n + x[1]


@ti.func
def v2m(x: int):
    return (int(x / num_node), int(x % num_node))


@ti.kernel
def init_m():
    for i in range(num_node):
        M[i, i] = 1.0


@ti.kernel
def update_force():

    for i in ti.grouped(x):
        idx = m2v(i)
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                F[idx] += -spring_Y * d * (current_dist / original_dist - 1)


@ti.kernel
def update_Jacobi():
    for i, d in J:
        J[i, d] *= 0.0
        idx_i = v2m(i)
        for offset in ti.static(spring_offsets):
            idx_j = idx_i + offset
            j = m2v(idx_j)
            if d == i or d == j:
                x_ij = x[idx_i] - x[idx_j]
                dir = x_ij.normalized()
                mat = dir.outer_product(dir)
                r = quad_size * float(idx_i - idx_j).norm()
                l = x_ij.norm()
                jx = -(spring_Y / r) * ((1 - r / l) * (I - mat) + mat)
                if d == i:
                    J[i, d] += jx
                else:
                    J[i, d] -= jx


@ti.kernel
def update_A():
    for i, j in A:
        A[i, j] = M[i, j] * I - dt * dt * J[i, j]


@ti.kernel
def update_b():
    for i in b:
        b[i] = v[v2m(i)] + dt * F[i]


@ti.kernel
def Jacobi_iteration():
    n = num_node
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] @ v[v2m(j)]
        new_v[i] = A[i, i].inverse() @ r
    for i in ti.grouped(v):
        v[i] = new_v[m2v(i)]


def implicit_substep(iter_number):
    init_m()
    update_Jacobi()
    update_A()
    update_force()
    update_b()
    for _ in range(iter_number):
        Jacobi_iteration()
    collision_detection()


@ti.kernel
def update_velocity():
    for i in ti.grouped(v):
        v[i] += F[m2v(i)] * dt


def explicit_substep():
    update_force()
    update_velocity()
    collision_detection()


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


window = ti.ui.Window("Cloth Simulation", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

# dump to images
# result_dir = "../video"
# video_manager = ti.tools.VideoManager(output_dir=result_dir,
#   framerate=24,
#   automatic_build=True)

#rendering loop
while window.running:
    if current_t > 3:
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        if use_implicit:
            implicit_substep(iter_number)
        else:
            explicit_substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    scene.particles(ball_center,
                    radius=ball_radius * 0.8,
                    color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    # video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()
