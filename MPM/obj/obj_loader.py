import sys
import tinyobjloader

# Create reader.
reader = tinyobjloader.ObjReader()

filename = "../model/bunny.obj"

# Load .obj(and .mtl) using default configuration
ret = reader.ParseFromFile(filename)

if ret == False:
    print("Warn:", reader.Warning())
    print("Err:", reader.Error())
    print("Failed to load : ", filename)

    sys.exit(-1)

if reader.Warning():
    print("Warn:", reader.Warning())

attrib = reader.GetAttrib()

print(attrib.vertices[0])
print(attrib.vertices[1])
print(attrib.vertices[2])

print("attrib.vertices = ", len(attrib.vertices))
print("attrib.normals = ", len(attrib.normals))
print("attrib.texcoords = ", len(attrib.texcoords))

materials = reader.GetMaterials()
print("Num materials: ", len(materials))
for m in materials:
    print(m.name)
    print(m.diffuse)

shapes = reader.GetShapes()

print(shapes[0].mesh.indices[0].vertex_index)
print(shapes[0].mesh.indices[1].vertex_index)
print(shapes[0].mesh.indices[2].vertex_index)

print("Num shapes: ", len(shapes))
for shape in shapes:
    print(shape.name)
    print("num_indices = {}".format(len(shape.mesh.indices)))