from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation
from multiprocessing import Process, Event
from pybullet_utils import bullet_client
from typing import Tuple
import numpy as np
import pybullet
import torch
import os
import sys
sys.path.append('/home/user/Workspace/imm-gym/immgymenvs/algos')
import franka_ik


RUN = 1
TERMINATE = -1

HAND = 8
LEFT_FINGER = 9
RIGHT_FINGER = 10

def _load_objects(sim):
    # Load objects
    # Plane
    planeId = sim.loadURDF("plane.urdf")
    # Panda
    startPos = [0, 0, 0]
    startOrientation = sim.getQuaternionFromEuler([0, 0, 0])
    robotId = sim.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=1)
    numJoints = sim.getNumJoints(robotId)
    # Table
    tableShape = (0.2, 0.25, 0.2)
    tablePosition = (0.5, 0.0, 0.2)
    tableVisualShapeId = sim.createVisualShape(
        shapeType=pybullet.GEOM_BOX,
        halfExtents=tableShape
    )
    tableCollisionShapeId = sim.createCollisionShape(
        shapeType=pybullet.GEOM_BOX, 
        halfExtents=tableShape
    )
    tableId = sim.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=tableCollisionShapeId,
        baseVisualShapeIndex=tableVisualShapeId,
        basePosition=tablePosition
    )
    # Card
    cardShape = (0.025, 0.035, 0.0025)
    cardPosition = (0.5, 0.0, 0.4025)
    cardColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
    cardVisualShapeId = sim.createVisualShape(
        shapeType=pybullet.GEOM_BOX,
        halfExtents=cardShape,
        rgbaColor=cardColor
    )
    cardCollisionShapeId = sim.createCollisionShape(
        shapeType=pybullet.GEOM_BOX, 
        halfExtents=cardShape
    )
    cardId = sim.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=cardCollisionShapeId,
        baseVisualShapeIndex=cardVisualShapeId,
        basePosition=cardPosition
    )
    return robotId, numJoints, cardId

def _set_joint_position(sim, robotId, numJoints, jointPosition):
    j = 0
    for i in range(numJoints):
        jointType = sim.getJointInfo(robotId, i)[2]
        if jointType == pybullet.JOINT_REVOLUTE:
            sim.resetJointState(robotId, i, jointPosition[j])
            j += 1
    sim.resetJointState(robotId, LEFT_FINGER, jointPosition[-2])
    sim.resetJointState(robotId, RIGHT_FINGER, jointPosition[-1])

# def _solve_inverse_kinematics(index: int, gripperConf: np.ndarray, q7: np.ndarray, jointPosition: np.ndarray):
#     initialJointPosition = np.array([0.0000, 0.0000, 0.0000, -0.9425, 0.0000, 1.1205, 0.0000, 0.0120, 0.0120])
#     jointPosition[index, :-3] = franka_ik.franka_IKCC(
#         gripperConf[index, :3], gripperConf[index, 3:7], q7[index], initialJointPosition[:-2]
#     )
#     jointPosition[index, -3:-1] = gripperConf[index, -1]
#     jointPosition[index, -1] = 1 - int(np.any(np.isnan(jointPosition[index, :-3]))) # Feasibility in terms of inverse kinematics

def _check_collision(sim, robotId):
    """
    Caution: This function returns True if the given gripper configuration is in collision with the object or the environment.
    """
    collision = False
    contactPoints = sim.getContactPoints(robotId)
    for i in range(len(contactPoints)):
        linkIndex = contactPoints[i][3]
        contactDistance = contactPoints[i][8]
        if linkIndex <= HAND and contactDistance < 0:
            collision = True
            break
        if linkIndex in [LEFT_FINGER, RIGHT_FINGER] and contactDistance < -0.005:
            collision = True
            break
    return collision

def mapActionToJointPositionWorker(
    run, done, commandShared, numWorkers: int, workerIndex: int, 
    batchSize: int, startIndex: int, numSamples: int, 
    gripperConfShared, q7Shared, initialObjectPoseShared, jointPositionShared
):
    import pybullet_data
    import pybullet

    command = np.ndarray(numWorkers, dtype=np.int8, buffer=commandShared.buf)
    gripperConf = np.ndarray((batchSize, 8), dtype=np.float64, buffer=gripperConfShared.buf)
    q7 = np.ndarray(batchSize, dtype=np.float64, buffer=q7Shared.buf)
    initialObjectPose = np.ndarray((batchSize, 7), dtype=np.float64, buffer=initialObjectPoseShared.buf)
    jointPosition = np.ndarray((batchSize, 10), dtype=np.float64, buffer=jointPositionShared.buf)

    sim = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotId, numJoints, objectId = _load_objects(sim)

    done.set() # Signal that initialization is complete

    while True:
        run.wait() # Wait for run signal
        run.clear()
        if command[workerIndex] == TERMINATE: break # Terminate the process
        for i in range(numSamples):
            currentIndex = (startIndex + i)
            jointPosition[currentIndex] = franka_ik.franka_IK(
                gripperConf[currentIndex, :3], gripperConf[currentIndex, 3:7], gripperConf[currentIndex, -1]
            )
            # _solve_inverse_kinematics(currentIndex, gripperConf, q7, jointPosition)
            sim.resetBasePositionAndOrientation(
                objectId, initialObjectPose[currentIndex, :3], initialObjectPose[currentIndex, 3:]
            )
            _set_joint_position(sim, robotId, numJoints, jointPosition[currentIndex])
            sim.performCollisionDetection()
            jointPosition[currentIndex, -1] *= (1 - int(_check_collision(sim, robotId)))
        done.set() # Signal that the job is done

    sim.disconnect()


class MapActionToJointPosition:
    def __init__(self, batchSize: int, numWorkers: int, rotationMatrix: bool = False):
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.rotationMatrix = rotationMatrix

        # Load pointcloud
        pointCloudDir = os.path.expanduser(
            '~/Workspace/ContactPoseSampler/pointcloud'
        )
        self.pointOnFinger = np.load(os.path.join(pointCloudDir, 'finger_points.npy'))
        self.pointOnObject = np.load(os.path.join(pointCloudDir, 'card_pointcloud_32.npy'))

        # Events to communicate with worker processes
        self.runs = list()
        for _ in range(self.numWorkers): self.runs.append(Event())
        self.dones = list()
        for _ in range(self.numWorkers): self.dones.append(Event())

        # Make shared memory used to communicate with worker processes
        self.smm = SharedMemoryManager()
        self.smm.start()
        self.commandShared, self.command = self._make_shared_memory(numWorkers, np.int8)
        self.gripperConfShared, self.gripperConf = self._make_shared_memory(
            (self.batchSize, 8), np.float64
        )
        self.q7Shared, self.q7 = self._make_shared_memory(self.batchSize, np.float64)
        self.initialObjectPoseShared, self.initialObjectPose = self._make_shared_memory(
            (self.batchSize, 7), np.float64
        )
        self.jointPositionShared, self.jointPosition = self._make_shared_memory(
            (self.batchSize, 10), np.float64
        )

        # Create pybullet workers
        self.workers = list()
        startIndices, jobAllocation = self._allocate_jobs()
        for i in range(numWorkers):
            worker = Process(
                target=mapActionToJointPositionWorker,
                args=(
                    self.runs[i], self.dones[i], self.commandShared, self.numWorkers, i,
                    self.batchSize, startIndices.item(i), jobAllocation.item(i), 
                    self.gripperConfShared, self.q7Shared, self.initialObjectPoseShared, self.jointPositionShared
                )
            )
            worker.start()
            self.workers.append(worker)
        self._wait_workers() # Wait until the initialization is complete

    def _make_shared_memory(self, shape, dtype):
        arrayTemp = np.zeros(shape, dtype=dtype)
        arrayShared = self.smm.SharedMemory(size=arrayTemp.nbytes)
        array = np.ndarray(arrayTemp.shape, arrayTemp.dtype, arrayShared.buf)
        array[:] = arrayTemp[:]
        return arrayShared, array

    def _allocate_jobs(self) -> Tuple[np.ndarray, np.ndarray]:
        jobAllocation = np.full(self.numWorkers, (self.batchSize / self.numWorkers), dtype=np.int32)
        remainingJobs = np.zeros_like(jobAllocation)
        remainingJobs[:(self.batchSize % self.numWorkers)] = 1
        jobAllocation += remainingJobs
        startIndices = np.roll(np.cumsum(jobAllocation), 1)
        startIndices[0] = 0
        return startIndices, jobAllocation

    def convert(self, action, initialObjectPose: torch.Tensor) -> torch.Tensor:
        device = initialObjectPose.device
        action, initialObjectPose = self._to_numpy(action, initialObjectPose)
        handRotation, distance, fingerContactPointL, fingerInContact, objectContactPointL = self._parse_action(action)
        objectContactPointW = self._to_world_coordinate(objectContactPointL, initialObjectPose)
        fingerPosition = self._get_finger_position(distance, fingerInContact)
        handPosition = self._get_hand_position(
            handRotation, fingerContactPointL, fingerPosition, objectContactPointW
        )
        self._set_gripper_conf(handPosition, handRotation, distance)
        # self._set_q7(q7)
        self._set_initial_object_pose(initialObjectPose)
        self._run_workers()
        self._wait_workers()
        jointPositionAndFeasibility = self._get_joint_position_and_feasibility(device)
        return jointPositionAndFeasibility

    def _to_numpy(self, action: torch.Tensor, initialObjectPose: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if type(action) is tuple: action = torch.cat(action, dim=1) # Make action a single tensor
        action = action.detach().cpu().numpy()
        initialObjectPose = initialObjectPose.detach().cpu().numpy()
        return action, initialObjectPose

    def _parse_action(self, action: np.ndarray) -> Tuple[Rotation, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.rotationMatrix:
            handRotation = Rotation.from_matrix(
                np.reshape(action[:, :9], (-1, 3, 3)).transpose((0, 2, 1))
            )
        else:
            handRotation = Rotation.from_quat(action[:, :4])
        if action.shape[-1] > 12: # Takes logits
            distance = np.clip(self._scale_action(action[:, -41], 0.0, 0.04), 0.0, 0.04)
            # q7 = np.clip(self._scale_action(action[:, -41], -2.8973, 2.8973), -2.8973, 2.8973)
            fingerContactIndex = np.argmax(action[:, -40:-32], axis=1)
            objectContactIndex = np.argmax(action[:, -32:], axis=1)
        else: # Takes indices
            distance = np.clip(self._scale_action(action[:, -3], 0.0, 0.04), 0.0, 0.04)
            # q7 = np.clip(self._scale_action(action[:, -3], -2.8973, 2.8973), -2.8973, 2.8973)
            fingerContactIndex = action[:, -2].astype(np.int32)
            objectContactIndex = action[:, -1].astype(np.int32)
        numPointsOnFinger = len(self.pointOnFinger)
        fingerInContact = (fingerContactIndex >= numPointsOnFinger).astype(int) # Which finger is in contact with the object?
        fingerContactIndex -= (fingerInContact * numPointsOnFinger)
        fingerContactPointL = self.pointOnFinger[fingerContactIndex]
        fingerContactPointL[:, :-1] *= np.expand_dims(
            (1 - fingerInContact * 2), axis=-1
        ) # In case of right finger, rotate 180 degrees w.r.t z axis
        objectContactPointL = self.pointOnObject[objectContactIndex]
        return handRotation, distance, fingerContactPointL, fingerInContact, objectContactPointL

    def _scale_action(self, action: np.ndarray, minValue: float, maxValue: float) -> np.ndarray:
        return (maxValue - minValue) / 2.0 * action + (maxValue + minValue) / 2.0

    def _to_world_coordinate(self, objectContactPointL: np.ndarray, initialObjectPose: np.ndarray) -> np.ndarray:
        objectPosition = initialObjectPose[:, :3]
        objectRotation = Rotation.from_quat(initialObjectPose[:, 3:])
        objectContactPointW = objectRotation.apply(objectContactPointL) + objectPosition
        return objectContactPointW

    def _get_finger_position(self, distance: np.ndarray, fingerInContact: np.ndarray) -> np.ndarray:
        """
        This function returns the position of finger frame w.r.t hand frame.
        """
        fingerPosition = np.zeros((len(distance), 3), dtype=distance.dtype)
        fingerPosition[:, 1] = distance
        fingerPosition[:, 2] = 0.0584
        fingerPosition[:, 1] *= (-fingerInContact) # In case of right finger, move finger backwards
        return fingerPosition

    def _get_hand_position(
        self, handRotation: Rotation, fingerContactPointL: np.ndarray, 
        fingerPosition: np.ndarray, objectContactPointW: np.ndarray
    ) -> np.ndarray:
        handPosition = objectContactPointW - handRotation.apply(fingerContactPointL + fingerPosition)
        return handPosition

    def _set_gripper_conf(self, handPosition: np.ndarray, handRotation: Rotation, distance: np.ndarray):
        self.gripperConf[:, :3] = handPosition
        self.gripperConf[:, 3:7] = handRotation.as_quat()
        self.gripperConf[:, -1] = distance

    def _set_q7(self, q7: np.ndarray):
        self.q7[:] = q7[:]

    def _set_initial_object_pose(self, initialObjectPose: np.ndarray):
        self.initialObjectPose[:] = initialObjectPose[:]

    def _get_joint_position_and_feasibility(self, device) -> torch.Tensor:
        if device == 'cpu':
            return torch.Tensor(self.jointPosition, device=device)
        else:
            return torch.from_numpy(self.jointPosition).to(device=device, dtype=torch.float)

    def _run_workers(self):
        self.command[:] = RUN
        for i in range(self.numWorkers): self.runs[i].set()
        
    def _wait_workers(self):
        for i in range(self.numWorkers): 
            self.dones[i].wait()
            self.dones[i].clear()
        
    def close(self):
        self.command[:] = TERMINATE
        for i in range(self.numWorkers): self.runs[i].set()
        for worker in self.workers:
            while worker.is_alive(): pass
        self.smm.shutdown()


def _generate_random_pose(numSamples: int) -> np.ndarray:
    """
    Generates random card poses. Do not use this function outside of this file.
    """
    initialObjectPosition = np.array([0.5, 0.0, 0.4025])
    initialObjectOrientation = np.array([0.0, 0.0, 0.0, 1.0])
    rng = np.random.default_rng()
    cardLength = np.sqrt(0.025**2 + 0.035**2)
    randomObjectPosition = np.zeros((numSamples, 3))
    randomObjectPosition[:, 0] = rng.uniform(
        (cardLength - 0.2),
        (0.2 - cardLength),
        numSamples
    )
    randomObjectPosition[:, 1] = rng.uniform(
        (cardLength - 0.25),
        (0.25 - cardLength),
        numSamples
    )
    randomObjectPosition += initialObjectPosition
    initialObjectRotation = Rotation.from_quat(initialObjectOrientation)
    randomObjectRotation = initialObjectRotation * Rotation.from_rotvec(
        np.outer(
            rng.uniform(0.0, (np.pi * 2), numSamples),
            np.array([0, 0, 1]) # rotate the object along the z axis
        )
    )
    randomObjectOrientation = randomObjectRotation.as_quat()
    randomObjectPose = np.concatenate((randomObjectPosition, randomObjectOrientation), axis=1).astype(np.float32)
    return randomObjectPose


if __name__ == "__main__":
    from supervised import PreContactPolicy
    from pathlib import Path
    import time

    batchSize = 10
    numWorkers = 2
    
    converter = MapActionToJointPosition(batchSize, numWorkers, True)

    # Load pre-contact policy
    preContactPolicy = PreContactPolicy([512, 256, 256, 128]).to('cuda:0')
    checkpointName = '2022-3-19-11-21.pth'
    checkpointPath = Path(os.path.dirname(__file__)).parent.joinpath('trains/pre-contact-policy/' + checkpointName)
    checkpoint = torch.load(checkpointPath)
    preContactPolicy.load_state_dict(checkpoint['model_state_dict'])
    preContactPolicy.eval()

    initialObjectPose = torch.tensor(_generate_random_pose(batchSize)).to('cuda:0')
    goalObjectPose = torch.tensor(_generate_random_pose(batchSize)).to('cuda:0')
    x = torch.cat((initialObjectPose, goalObjectPose), dim=1)
    action = preContactPolicy(x)
    t = time.time()
    jointPosition = converter.convert(action, initialObjectPose)
    print(f'{(time.time() - t) * 1000:.2f}ms')
    print(torch.count_nonzero(jointPosition[:, -1]))
    converter.close()
    jointPositionNumpy = jointPosition.cpu().numpy()
    jointPositionDir = os.path.expanduser(
        '~/Workspace/ContactPoseSampler/pointcloud'
    )
    initialObjectPoseNumpy = initialObjectPose.cpu().numpy()
    initialObjectPoseDir = os.path.expanduser(
        '~/Workspace/ContactPoseSampler/pointcloud'
    )
    np.save(os.path.join(jointPositionDir, 'joint_position.npy'), jointPositionNumpy)
    np.save(os.path.join(initialObjectPoseDir, 'initial_object_pose.npy'), initialObjectPoseNumpy)
    