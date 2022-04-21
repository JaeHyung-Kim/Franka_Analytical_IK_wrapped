from scipy.spatial.transform import Rotation
import pybullet_data
import numpy as np
import franka_ik
import pybullet
import trimesh
import utils
import time


# Initialization
t = time.time()
physicsClient = pybullet.connect(pybullet.DIRECT)
dataPath = pybullet_data.getDataPath()
pybullet.setAdditionalSearchPath(dataPath)

valids = []
robotId, numJoints, objectId = utils.loadObjects()

HAND = 8
LEFT_FINGER = 9
RIGHT_FINGER = 10

# Load meshes
finger = trimesh.load(dataPath + "/franka_panda/meshes/collision/finger.obj")
card = trimesh.primitives.Box(extents=(0.05, 0.07, 0.005))

NUM_SAMPLES = 20
SAMPLES_BUF = NUM_SAMPLES * 100
fingerInContact = LEFT_FINGER

# Randomize the object pose
randomObjectPosition, randomObjectRotation = utils.randomizeObjectPose(objectId, SAMPLES_BUF)

# Sample random contact points on the surface of the finger
difference = np.linalg.norm(
    (finger.face_normals - np.array([0, 0, 1])),
    axis=-1
)
valid_face_indices = np.argpartition(difference, 2)[:2]
valid_vertex_indices = np.unique(finger.faces[valid_face_indices])
valid_vertices = np.unique(
    finger.vertices[valid_vertex_indices],
    axis=0
)
vertex_indices = np.random.default_rng().choice(
    valid_vertices.shape[0],
    SAMPLES_BUF
)
contactPointOnFinger = valid_vertices[vertex_indices]
# For right finger
if fingerInContact == RIGHT_FINGER:
    contactPointOnFinger[:, :-1] *= -1 # rotate 180 degrees w.r.t z axis

# Sample random contact points on the surface of the object(card)
weight = np.prod(
    (card.face_normals == np.array([0, 0, 1])), 
    axis=-1
)
weight = weight.astype(float) / np.count_nonzero(weight)
contactPointOnObject, cardFaceIndex = card.sample(SAMPLES_BUF, return_index=True, face_weight=weight)
contactNormalOnObject = card.face_normals[cardFaceIndex]

# Random contact points on the object w.r.t world
contactPointOnObject = randomObjectRotation.apply(contactPointOnObject) + randomObjectPosition
contactNormalOnObject = randomObjectRotation.apply(contactNormalOnObject)

# Sample distance between fingers randomly
distance, fingerPositionToHand = utils.randomizeDistanceBetweenFingers(
    SAMPLES_BUF, fingerInContact
)

# Set target hand position and orientation
fingerRotation = Rotation.random(SAMPLES_BUF)
fingerRotationMatrix = fingerRotation.as_matrix()
zAxis = fingerRotationMatrix[..., 2]
mask = np.sum((zAxis * contactNormalOnObject), axis=-1) >= 0
correctedFingerRotation = fingerRotation * Rotation.from_rotvec((np.pi, 0, 0))
fingerRotation = Rotation.from_matrix(
    np.expand_dims((1 - mask), axis=(1, 2)) * fingerRotationMatrix 
    + np.expand_dims(mask, axis=(1, 2)) * correctedFingerRotation.as_matrix()
)
fingerPosition = contactPointOnObject - fingerRotation.apply(contactPointOnFinger)
handOrientation = fingerRotation.as_quat()
handPosition = fingerPosition - fingerRotation.apply(fingerPositionToHand)

# Initialze joint position
initialJointPosition = np.array([0.0000, 0.0000, 0.0000, -0.9425, 0.0000, 1.1205, 0.0000, 0.0120, 0.0120])
lowerLimits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
upperLimits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

randomObjectOrientation = randomObjectRotation.as_quat()

sum1 = 0.0
sum2 = 0.0
Q7_CANDIDATES = 100
q7_candidates = np.linspace(-2.8973, 2.8973, (Q7_CANDIDATES + 2))[1:-1]
indicator = np.zeros(Q7_CANDIDATES, np.int32)
feasibility = np.zeros(Q7_CANDIDATES, np.int32)
solutions = np.zeros((Q7_CANDIDATES, 7), np.float64)
for i in range(NUM_SAMPLES):
    # Set position and orientation of the object
    # pybullet.resetBasePositionAndOrientation(objectId, randomObjectPosition[i], randomObjectOrientation[i])
    # Set joint position

    t1 = time.time()
    utils.setJointPosition(robotId, numJoints, initialJointPosition)
    jointPosition = np.array(pybullet.calculateInverseKinematics(
        bodyUniqueId=robotId, 
        endEffectorLinkIndex=HAND, 
        targetPosition=handPosition[i],
        targetOrientation=handOrientation[i],
        maxNumIterations=1000,
        residualThreshold=1e-6
    ))
    jointPosition[-2:] = distance[i]
    sum1 += time.time() - t1
    index = np.argmin(np.abs(q7_candidates - jointPosition[6]))
    indicator[index] = 1
    print(indicator)
    indicator[index] = 0
    # utils.setJointPosition(robotId, numJoints, jointPosition)

    t2 = time.time()
    jointPositionAnalytical = franka_ik.franka_IK(
        handPosition[i], handOrientation[i], distance[i]
    )
    print(jointPositionAnalytical)
    # for j in range(Q7_CANDIDATES):
    #     jointPositionAnalytical = franka_ik.franka_IK_q7(
    #         handPosition[i], handOrientation[i], distance[i], q7_candidates[j], 
    #     )
    #     feasibility[j] = int(jointPositionAnalytical[-1])
    #     if feasibility[j] == 1: solutions[j] = jointPositionAnalytical[:-3]
    #     else: solutions[j] = 100
    #     if feasibility[j] == 0: continue
    #     utils.setJointPosition(robotId, numJoints, jointPositionAnalytical[:-1])
    #     pybullet.performCollisionDetection()
    #     print(jointPositionAnalytical[:-1])
    #     time.sleep(2)
    sum2 += time.time() - t2
    # utils.setJointPosition(robotId, numJoints, jointPositionAnalytical[:-1])
    # print(feasibility)
    # index = np.argmin(np.linalg.norm((initialJointPosition[:-2] - solutions), axis=-1))
    if jointPositionAnalytical[-1] == 1:
        index = np.argmin(np.abs(q7_candidates - jointPositionAnalytical[6]))
        indicator[index] = 1
    print(indicator)
    indicator[index] = 0
    # pybullet.performCollisionDetection()
    # time.sleep(10)
    # print(jointPositionAnalytical)
    # print(jointPositionAnalytical.flags['OWNDATA'])
    # print(jointPositionAnalytical.flags['WRITEABLE'])
    print("==========")
    

print("Numerical IK: " + f'{(sum1) * 1000 * 1000 / NUM_SAMPLES}μs')
print("Analytical IK: " + f'{(sum2) * 1000 * 1000 / NUM_SAMPLES}μs')

pybullet.disconnect()
