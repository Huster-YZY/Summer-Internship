import numpy as np
from plyfile import PlyData


class PlyImporter:
    def __init__(self, file):
        self.file_name = file
        plydata = PlyData.read(self.file_name)
        data = plydata['vertex'].data
        self.count = plydata['vertex'].count
        l = len(data[0])
        if l == 3:
            self.np_array = np.array([[x, y, z] for x, y, z in data])
        else:
            t = np.array([[x, y, z] for x, y, z, _, _, _ in data],
                         dtype=np.float32)
            self.np_array = t + 0.5

    def get_array(self):
        return self.np_array

    def get_count(self):
        return self.count

    def multiply(self, mul):
        self.np_array *= mul


if __name__ == "__main__":
    ply1 = PlyImporter("/Users/YZY/g201/MPM/model/bunny.ply")
    print(ply1.get_count())
    ply1 = PlyImporter("/Users/YZY/g201/MPM/frames/mpm3d_000000.ply")
    print(ply1.get_count())
