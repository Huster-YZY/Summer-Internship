#classical opening
import taichi as ti

ti.init(arch=ti.metal)

#configure parameters
quality = 1
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1.0 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rou = (0.5 * dx)**2, 1
p_mass = p_vol * p_rou
E, nu = 0.1e4, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
down_boundary = 2.5e-2
up_boundary = 4.5e-3
# global variable
x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])


@ti.func
def P2G():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (I + dt * C[p]) @ F[p]
        h = ti.exp(10 * (1 - Jp[p]))

        if material[p] == 1:  #jelly
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:
                new_sig = ti.min(ti.max(sig[d, d], 1 - down_boundary),
                                 1 + up_boundary)
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
            J *= new_sig

        # process different material
        if material[p] == 0:  #fluid
            F[p] = I * ti.sqrt(J)
        elif material[p] == 2:  #snow
            F[p] = U @ sig @ V.transpose()

        #MLS-MPM (need to be learned)
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        #p2g
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            idx = base + offset
            grid_v[idx] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[idx] += weight * p_mass


@ti.func
def boundary():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            # update velocity
            grid_v[i, j] = (1.0 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j][1] -= dt * 50

        if i < 3 and grid_v[i, j][0] < 0:
            grid_v[i, j][0] = 0
        if i > n_grid - 3 and grid_v[i, j][0] > 0:
            grid_v[i, j][0] = 0
        if j < 3 and grid_v[i, j][1] < 0:
            grid_v[i, j][1] = 0
        if j > n_grid - 3 and grid_v[i, j][1] > 0:
            grid_v[i, j][1] = 0


@ti.func
def G2P():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += v[p] * dt


@ti.kernel
def substep():
    #initialize euler grid
    for i, j in grid_m:
        grid_m[i, j] = 0.0
        grid_v[i, j] = ti.Vector([0, 0])
    P2G()
    boundary()
    G2P()


#we will simulate 3 materials including fluid,elastic object and snow
group_size = n_particles // 4


@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3 + 0.1 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.20 * (i // group_size)
        ]
        material[i] = 2  # i // group_size
        v[i] = ti.Vector([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1


def main():
    initialize()
    gui = ti.GUI("YZY-MPM", res=512, background_color=0x191970)
    while not gui.get_event(ti.GUI.ESCAPE):
        for _ in range(int(2e-3 // dt)):
            substep()
        gui.circles(x.to_numpy(),
                    radius=1,
                    palette=[0x068587, 0xED553B, 0xEEEEF0],
                    palette_indices=material)
        video_manager.write_frame(gui.get_image())
        gui.show()


result_dir = "../video"
video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                      framerate=60,
                                      automatic_build=True)
if __name__ == "__main__":
    main()
