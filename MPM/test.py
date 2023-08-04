import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

a = np.array([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
b = np.array([[0, 1, 2]])

print(ti.Vector(a[0]))