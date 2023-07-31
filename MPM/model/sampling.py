import open3d as o3d

path = './bunny.ply'
mesh = o3d.io.read_point_cloud(path)

mesh.compute_vertex_normals()

pcd = mesh.sample_points_uniformly(number_of_points=5000)

# o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud("./bunny.ply", pcd, write_ascii=True)
