import taichi as ti
import trimesh as tm  # load the triangle mesh
import numpy as np

# metal doesn't support f64 type
ti.init(arch=ti.metal)
import time


def main():

    # Voxelization***********************************************************************
    t0 = time.time()
    mesh = tm.load("./model/bunny.obj")
    voxelized_mesh = mesh.voxelized(pitch=0.002).fill()
    voxelized_points_np = voxelized_mesh.points.astype(np.float32)
    num_particles_obj = voxelized_points_np.shape[0]
    voxelized_points = ti.Vector.field(3, float, num_particles_obj)
    voxelized_points.from_numpy(voxelized_points_np)
    t1 = time.time()
    print("Particles Num:%d \t Time Use:%f" % (num_particles_obj, t1 - t0))
    #     np.savetxt("voxelized_points_np.csv",voxelized_points_np)

    #visualization***********************************************************************
    window = ti.ui.Window("Test voxelize", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 4)
    while window.running:
        camera.track_user_inputs(window,
                                 movement_speed=0.005,
                                 hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(voxelized_points,
                        color=(0.68, 0.26, 0.19),
                        radius=0.001)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()