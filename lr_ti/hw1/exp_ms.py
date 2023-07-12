import taichi as ti

ti.init(arch=ti.cpu)

# the scale of the simulation
n = 128
quad_size = 1.0 / n

dt = 3.125e-4
# dt = 4e-2 / n

# substeps=53,the number of iteration if we want to achieve 60fps
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e4
dashpot_damping = 1
drag_damping = 1
# assume each particle's mass is Identity matrix
# M = ti.Matrix.field(3, 3, float, (n, n))

J = ti.Matrix.field(3, 3, float, (n, n))

# balls' model
ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(3, ))
ball_center[0] = [0, 0, 0]
ball_center[1] = [0.4, 0, 0]
ball_center[2] = [-0.4, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# configuration
bending_springs = False
use_implicit = True


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


@ti.func
def clr(m):
    for i in range(3):
        for j in range(3):
            m[i, j] = 0.0


@ti.func
def Jacobi_iterations(A, idx, b):
    new_x = ti.Vector([0.0, 0.0, 0.0])

    for i in range(3):
        r = b[i]
        for j in range(3):
            if i != j:
                r -= A[i, j] * v[idx][j]
        new_x[i] = r / A[i, i]

    for i in range(3):
        v[idx][i] = new_x[i]


@ti.func
def collision_detection():
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        for j in range(3):
            offset_to_center = x[i] - ball_center[j]
            if offset_to_center.norm() <= ball_radius:
                normal = offset_to_center.normalized()
                v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += v[i] * dt


# Most Important Function


@ti.kernel
def exp_substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()

                force += -spring_Y * d * (current_dist / original_dist - 1)
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt

    # collision detection
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
        exp_substep()
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
