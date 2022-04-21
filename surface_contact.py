from scipy.spatial.transform import Rotation
from ctypes import CDLL
import pybullet_data
import numpy as np
import pybullet
import trimesh
import ctypes
import random
import utils
import time


libfrankaik = CDLL("/home/guest/Simulation/franka_analytical_ik/build/libfrankaik.so")

libfrankaik.frankaIK.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64), 
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_double, 
    np.ctypeslib.ndpointer(dtype=np.float64), 
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=2, shape=(4, 7), flags='C_CONTIGUOUS'
    )
]

libfrankaik.frankaIKCC.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64), 
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_double, 
    np.ctypeslib.ndpointer(dtype=np.float64), 
    np.ctypeslib.ndpointer(dtype=np.float64)
]

# Initialization
physicsClient = pybullet.connect(pybullet.DIRECT)
dataPath = pybullet_data.getDataPath()
pybullet.setAdditionalSearchPath(dataPath)

robotId, numJoints, objectId = utils.loadObjects()

HAND = 8
LEFT_FINGER = 9
RIGHT_FINGER = 10 

# handState = pybullet.getLinkState(robotId, HAND)
# handPosition = np.array(handState[4])
# handRotation = Rotation.from_quat(handState[5])

# leftFingerState = pybullet.getLinkState(robotId, LEFT_FINGER)
# leftFingerPosition = np.array(leftFingerState[4])
# leftFingerRotation = Rotation.from_quat(leftFingerState[5])

# rightFingerState = pybullet.getLinkState(robotId, RIGHT_FINGER)
# rightFingerPosition = np.array(rightFingerState[4])
# rightFingerRotation = Rotation.from_quat(rightFingerState[5])

# leftFingerToHandPosition = leftFingerRotation.apply((handPosition - leftFingerPosition), inverse=True)
# leftFingerToHandRotation = leftFingerRotation.inv() * handRotation
# print(leftFingerToHandPosition)
# print(leftFingerToHandRotation.as_matrix())
# print('----------')
# rightFingerToHandPosition = rightFingerRotation.apply((handPosition - rightFingerPosition), inverse=True)
# rightFingerToHandRotation = rightFingerRotation.inv() * handRotation
# print(rightFingerToHandPosition)
# print(rightFingerToHandRotation.as_matrix())
# print('----------')

# Load meshes
finger = trimesh.load(dataPath + "/franka_panda/meshes/collision/finger.obj")
card = trimesh.primitives.Box(extents=(0.05, 0.07, 0.005))

NUM_SAMPLES = 50
fingerInContact = LEFT_FINGER

# Randomize the object pose
randomObjectPosition, randomObjectRotation = utils.randomizeObjectPose(objectId, NUM_SAMPLES)

# Sample random contact points on the surface of the finger
difference = np.linalg.norm(
    (finger.face_normals - np.array([0, 0, 1])),
    axis=-1
)
indices = np.argpartition(difference, 2)[:2]
multiplier = np.zeros_like(finger.area_faces)
multiplier[indices] = 1
weight = finger.area_faces * multiplier
weight = weight / np.sum(weight)
contactPointOnFinger, fingerFaceIndex = finger.sample(NUM_SAMPLES, return_index=True, face_weight=weight)
contactNormalOnFinger = finger.face_normals[fingerFaceIndex]
# For right finger
if fingerInContact == RIGHT_FINGER:
    contactPointOnFinger[:, :-1] *= -1 # rotate 180 degrees w.r.t z axis
    contactNormalOnFinger[:, :-1] *= -1 # rotate 180 degrees w.r.t z axis

# Sample random contact points on the surface of the object(card)
weight = np.prod(
    (card.face_normals == np.array([0, 0, 1])), 
    axis=-1
)
weight = weight.astype(float) / np.count_nonzero(weight)
contactPointOnObject, objectFaceIndex = card.sample(NUM_SAMPLES, return_index=True, face_weight=weight)
contactNormalOnObject = card.face_normals[objectFaceIndex]

# Random contact points on the object w.r.t world
contactPointOnObject = randomObjectRotation.apply(contactPointOnObject) + randomObjectPosition
contactNormalOnObject = randomObjectRotation.apply(contactNormalOnObject)

# Sample distance between fingers randomly
distance, fingerPositionToHand = utils.randomizeDistanceBetweenFingers(
    NUM_SAMPLES, fingerInContact
)

# Set target hand position and orientation
fingerRotation = Rotation.from_matrix(utils.getRotationMatrix(contactNormalOnFinger, -contactNormalOnObject))
randomRotation = Rotation.from_rotvec(
    np.expand_dims(
        np.random.default_rng().uniform(0, (2 * np.pi), NUM_SAMPLES),
        axis=-1
    ) * contactNormalOnObject
)
randomFingerRotation = randomRotation * fingerRotation
fingerPosition = contactPointOnObject - randomFingerRotation.apply(contactPointOnFinger)
handOrientation = randomFingerRotation.as_quat()
handPosition = fingerPosition - randomFingerRotation.apply(fingerPositionToHand)

# Initialze joint position
initialJointPosition = np.array([0.0000, 0.0000, 0.0000, -0.9425, 0.0000, 1.1205, 0.0000, 0.0120, 0.0120])

randomObjectOrientation = randomObjectRotation.as_quat()

jointPositionAnalytical = np.zeros(9, dtype=np.float64)

t = time.time()
for i in range(NUM_SAMPLES):
    # Set position and orientation of the object
    pybullet.resetBasePositionAndOrientation(objectId, randomObjectPosition[i], randomObjectOrientation[i])

    # Solve numerical IK
    # utils.setJointPosition(robotId, numJoints, initialJointPosition)
    # jointPosition = np.array(pybullet.calculateInverseKinematics(
    #     bodyUniqueId=robotId, 
    #     endEffectorLinkIndex=HAND, 
    #     targetPosition=handPosition[i],
    #     targetOrientation=handOrientation[i],
    #     maxNumIterations=1000,
    #     residualThreshold=1e-6
    # ))
    # jointPosition[-2:] = distance[i]
    # print(jointPosition)

    # handRotation = Rotation.from_quat(handOrientation[i])
    # gripperPosition = handRotation.apply((0.0, 0.0, 0.1034)) + handPosition[i]
    # gripperOrientation = (handRotation * Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 2]))).as_quat()
    
    # libfrankaik.frankaIKCC(
    #     handPosition[i], handOrientation[i], 
    #     jointPosition[6], initialJointPosition[:-2], jointPositionAnalytical
    # )

    # Solve analytical IK
    while True:
        print(handPosition[i])
        libfrankaik.frankaIKCC(
            handPosition[i], handOrientation[i], 
            random.uniform(-2.8973, 2.8973), initialJointPosition[:-2], jointPositionAnalytical
        )
        if not np.any(np.isnan(jointPositionAnalytical)): break
    jointPositionAnalytical[7:] = distance[i]
    print(jointPositionAnalytical)

    # pybullet.performCollisionDetection()

    # infeasible = utils.checkFeasibility(robotId, handPosition[i], handOrientation[i])
    # if infeasible: print('infeasible')
    # else:
    #     print('feasible')
    #     utils.printContactPoints(robotId)
        # positionOnRobot = contactPoints[:][5]
        # print(positionOnRobot)

    # handStateIK = pybullet.getLinkState(robotId, HAND)
    # handPositionIK = np.array(handStateIK[4])
    # handRotationIK = Rotation.from_quat(handStateIK[5])
    # handRotation = Rotation.from_quat(handOrientation[i])
    # handToGripperPosition = handRotationIK.inv().apply(handPosition[i] - handPositionIK)
    # handToGripperRotation = handRotationIK.inv() * handRotation
    # print(handToGripperPosition)
    # print(handToGripperRotation.as_rotvec())
    # print("=====")
    # contactPointOnFingerIK = handRotationIK.apply(contactPointOnFinger[0] + fingerPositionToHand[0]) + handPositionIK
    # print(contactPointOnFingerIK)
print(f'{(time.time() - t) * 1000}ms')

pybullet.disconnect()