from scipy.spatial.transform import Rotation
import numpy as np
import pybullet


HAND = 8
LEFT_FINGER = 9
RIGHT_FINGER = 10

# def skew(x):
#     if (isinstance(x, np.ndarray) and len(x.shape)>=2):
#         return np.array([
#             [0, -x[2][0], x[1][0]],
#             [x[2][0], 0, -x[0][0]],
#             [-x[1][0], x[0][0], 0]
#         ])
#     else:
#         return np.array([
#             [0, -x[2], x[1]],
#             [x[2], 0, -x[0]],
#             [-x[1], x[0], 0]
#         ])

# def getRotationMatrix(n1, n2):
#     if np.all(n1 == n2):
#         return np.identity(3)
#     elif np.all(n1 == -n2):
#         return -np.identity(3)
#     else:
#         axis = np.cross(n1, n2)
#         axis /= np.linalg.norm(axis)
#         c = np.dot(n1, n2)
#         s = np.sqrt(1 - np.square(c))
#         w = skew(axis)
#         return np.identity(3) + s * w + (1 - c) * np.matmul(w, w)

def loadObjects():
    # Load objects
    # Plane
    planeId = pybullet.loadURDF("plane.urdf")
    # Panda
    startPos = [0, 0, 0]
    startOrientation = pybullet.getQuaternionFromEuler([0, 0, 0])
    robotId = pybullet.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=1)
    numJoints = pybullet.getNumJoints(robotId)
    # Table
    tableShape = (0.2, 0.25, 0.2)
    tablePosition = (0.5, 0.0, 0.2)
    tableVisualShapeId = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_BOX,
        halfExtents=tableShape
    )
    tableCollisionShapeId = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_BOX, 
        halfExtents=tableShape
    )
    tableId = pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=tableCollisionShapeId,
        baseVisualShapeIndex=tableVisualShapeId,
        basePosition=tablePosition
    )
    # Card
    cardShape = (0.025, 0.035, 0.0025)
    cardPosition = (0.5, 0.0, 0.4025)
    cardColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
    cardVisualShapeId = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_BOX,
        halfExtents=cardShape,
        rgbaColor=cardColor
    )
    cardCollisionShapeId = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_BOX, 
        halfExtents=cardShape
    )
    cardId = pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=cardCollisionShapeId,
        baseVisualShapeIndex=cardVisualShapeId,
        basePosition=cardPosition
    )
    return robotId, numJoints, cardId

def randomizeObjectPose(objectId, numSamples):
    initialObjectPosition, initialObjectOrientation = pybullet.getBasePositionAndOrientation(objectId)
    cardLength = np.sqrt(0.025**2 + 0.035**2)
    randomObjectPosition = np.zeros((numSamples, 3))
    randomObjectPosition[:, 0] = np.random.default_rng().uniform(
        (cardLength - 0.2),
        (0.2 - cardLength),
        numSamples
    )
    randomObjectPosition[:, 1] = np.random.default_rng().uniform(
        (cardLength - 0.25),
        (0.25 - cardLength),
        numSamples
    )
    randomObjectPosition += initialObjectPosition
    initialObjectRotation = Rotation.from_quat(initialObjectOrientation)
    randomObjectRotation = initialObjectRotation * Rotation.from_rotvec(
        np.outer(
            np.random.default_rng().uniform(0.0, (np.pi * 2), numSamples),
            np.array([0, 0, 1]) # rotate the object along the z axis
        )
    )
    return randomObjectPosition, randomObjectRotation

def getEdges(finger):
    difference = np.linalg.norm(
        (finger.face_normals - np.array([0, 0, 1])),
        axis=-1
    )
    bottom_face_indices = np.argpartition(difference, 2)[:2]
    bottom_vertex_indices = finger.faces[bottom_face_indices]
    indices1 = bottom_vertex_indices[0]
    indices2 = bottom_vertex_indices[1]
    vertices1 = finger.vertices[indices1]
    vertices2 = finger.vertices[indices2]
    bottom_vertices, counts = np.unique(
        np.concatenate((vertices1, vertices2), axis=0),
        return_counts=True,
        axis=0
    )
    unique_bottom_vertices = bottom_vertices[np.nonzero(counts == 1)[0]]
    common_vertex_indices1 = indices1[
        np.nonzero(
            np.logical_not(
                np.all(
                    np.logical_or(
                        (vertices1 == unique_bottom_vertices[0]),
                        (vertices1 == unique_bottom_vertices[1])
                    ),
                    axis=-1
                )
            )
        )[0]
    ]
    common_vertex_indices2 = indices2[
        np.nonzero(
            np.logical_not(
                np.all(
                    np.logical_or(
                        (vertices2 == unique_bottom_vertices[0]),
                        (vertices2 == unique_bottom_vertices[1])
                    ),
                    axis=-1
                )
            )
        )[0]
    ]
    bottom_edge_indices = finger.faces_unique_edges[bottom_face_indices].flatten()
    bottom_edges = finger.edges_unique[bottom_edge_indices]
    infeasible_edge_indices = bottom_edge_indices[
        np.nonzero(
            np.logical_or(
                np.all(
                    np.isin(bottom_edges, common_vertex_indices1),
                    axis=-1
                ),
                np.all(
                    np.isin(bottom_edges, common_vertex_indices2),
                    axis=-1
                )
            )
        )[0]
    ]
    valid_face_indices = np.argpartition(difference, 7)[:7]
    candidate_edge_indices = finger.faces_unique_edges[valid_face_indices].flatten()
    valid_edge_indices = np.setdiff1d(candidate_edge_indices, infeasible_edge_indices)
    valid_vertex_indices = finger.edges_unique[valid_edge_indices]
    valid_vertices = finger.vertices[valid_vertex_indices]
    valid_vertices_reversed = np.zeros_like(valid_vertices)
    valid_vertices_reversed[:, 0, :] = valid_vertices[:, 1, :]
    valid_vertices_reversed[:, 1, :] = valid_vertices[:, 0, :]
    mask = np.expand_dims(
        valid_vertices[:, 0, 0] < valid_vertices[:, 1, 0],
        axis=(1, 2)
    )
    valid_vertices_ordered = valid_vertices * mask + valid_vertices_reversed * (1 - mask)
    valid_vertices_ordered = np.reshape(valid_vertices_ordered, (-1, 6))
    unique_valid_vertices = np.unique(valid_vertices_ordered, axis=0)
    return np.reshape(unique_valid_vertices, (-1, 2, 3))

def skew(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    x0 = np.zeros_like(x1)
    r1 = np.stack([x0, -x3, x2], axis=-1)
    r2 = np.stack([x3, x0, -x1], axis=-1)
    r3 = np.stack([-x2, x1, x0], axis=-1)
    return np.stack([r1, r2, r3], axis=1)

def getRotationMatrix(v1, v2):
    w = np.cross(v1, v2)
    length = np.linalg.norm(w, axis=-1, keepdims=True)
    mask = (length < 1e-8)
    w = w / np.ma.MaskedArray(length, mask=mask)
    c = np.sum((v1 * v2), axis=-1)
    s = np.sqrt(1 - np.clip(np.square(c), 0, 1))
    w = skew(w)
    R = np.einsum('i,ijk->ijk', s, w)
    R += np.einsum('i,ijk->ijk', (1 - c), np.einsum('ijk,ikl->ijl', w, w))
    mask = np.squeeze(mask)
    for i in range(3):
        R[:, i, i] += (np.ones_like(c) * (1 - mask) + np.sign(c) * mask)
    return R

def randomizeDistanceBetweenFingers(numSamples, fingerInContact):
    distance = np.random.default_rng().uniform(0.0, 0.04, numSamples)
    fingerPositionToHand = np.zeros((numSamples, 3))
    fingerPositionToHand[:, 1] = distance
    fingerPositionToHand[:, 2] = 0.0584
    if fingerInContact == RIGHT_FINGER:
        fingerPositionToHand[:, 1] *= -1 # move finger backwards
    return distance, fingerPositionToHand

def setJointPosition(robotId, numJoints, jointPosition):
    j = 0
    for i in range(numJoints):
        jointType = pybullet.getJointInfo(robotId, i)[2]
        if jointType == pybullet.JOINT_REVOLUTE:
            pybullet.resetJointState(robotId, i, jointPosition[j])
            j += 1
    pybullet.resetJointState(robotId, LEFT_FINGER, jointPosition[-2])
    pybullet.resetJointState(robotId, RIGHT_FINGER, jointPosition[-1])

def checkFeasibility(robotId, targetHandPosition, targetHandOrientation):
    # Inverse Kinematics
    threshold = 1e-5
    handState = pybullet.getLinkState(robotId, HAND)
    handPositionIK = np.array(handState[4])
    handOrientationIK = np.array(handState[5])
    positionError = np.linalg.norm((targetHandPosition - handPositionIK))
    if positionError > threshold:
        return True
    orientationError = min(
        np.linalg.norm((targetHandOrientation - handOrientationIK)),
        np.linalg.norm((targetHandOrientation + handOrientationIK))
    )
    if orientationError > threshold:
        return True

    # Collision
    infeasible = False
    contactPoints = pybullet.getContactPoints(robotId)
    for i in range(len(contactPoints)):
        linkIndex = contactPoints[i][3]
        contactDistance = contactPoints[i][8]
        if linkIndex <= HAND and contactDistance < 0:
            infeasible = True
            break
        if linkIndex in [LEFT_FINGER, RIGHT_FINGER] and contactDistance < -0.005:
            infeasible = True
            break
    return infeasible

def printContactPoints(robotId: int) -> None:
    contactPointsOnLeftFinger = pybullet.getContactPoints(bodyA=robotId, linkIndexA=LEFT_FINGER)
    print(f'left: {len(contactPointsOnLeftFinger)}')
    contactPointsOnRightFinger = pybullet.getContactPoints(bodyA=robotId, linkIndexA=RIGHT_FINGER)
    print(f'right: {len(contactPointsOnRightFinger)}')
