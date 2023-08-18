import pybullet as p
import time
import pybullet_data

#set up a GUI
physicsClient = p.connect(p.GUI)

p.setGravity(0, 0, -10)
planeId = p.loadURDF("./model/plane/plane.urdf")

startPos = [0, 0, 2]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("./model/block/block.urdf", startPos, startOrientation)

for i in range(100):
    p.stepSimulation()
    time.sleep(1. / 240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()