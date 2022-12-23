import os
import gym
import pickle
import pybullet as pb
import pybullet_data
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
import time

def quatInvQuatMul(q, p):
    pq = np.copy(p)

    q = q[[3, 0, 1, 2]]
    p = p[[3, 0, 1, 2]]

    pq[0] = p[0] * q[0] - p[1] * -q[1] - p[2] * -q[2] - p[3] * -q[3]
    pq[1] = p[0] * -q[1] + p[1] * q[0] - p[2] * -q[3] + p[3] * -q[2]
    pq[2] = p[0] * -q[2] + p[1] * -q[3] + p[2] * q[0] - p[3] * -q[1]
    pq[3] = p[0] * -q[3] - p[1] * -q[2] + p[2] * -q[1] + p[3] * q[0]

    return pq[[3, 0, 1, 2]]

class LegoEnv(gym.Env):
    # metadata = {'render.modes': ['human']}  

    def __init__(self):
        # wrist translation(3) + wrist rotation(3) + joint rotation(45) = 51
        self.action_space = gym.spaces.box.Box(low=np.array([-1]*2+[0]+[-np.pi]*48),
                                                high=np.array([1]*2+[1]+[np.pi]*48))

        # wrist translation(3) + wrist rotation(3) + joint rotation(45) + lego translation(3) + lego rotation(3) = 57
        # joint angles(45) + joint angular velocities(45) + each joint forces exceeded on object(21) + 6D pose of the wrist(6) +
        # 6D pose of the object(6) + velocity of the wrist pose(6) + velocity of the object pose(6)
        self.observation_space = gym.spaces.box.Box(low=np.array([-1]*2+[0]+[-np.pi]*48+[-1]*3+[-np.pi]*3),
                                                    high=np.array([1]*2+[1]+[np.pi]*48+[1]*3+[np.pi]*3))

        # connect to pybullet
        pb.connect(pb.GUI)
        pb.setTimeStep(1 / 60)
        # pb.resetSimulation()
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)

        # set plane and table
        # pb.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.plane_id = pb.loadURDF("plane.urdf")

        # add mano urdf
        self.mano_id = pb.loadURDF("/manoUrdf/Leo/mano.urdf", [0, 0, 0], pb.getQuaternionFromEuler([0, 0, 0]))
        self.available_joints_indexes = [i for i in range(pb.getNumJoints(self.mano_id)) if pb.getJointInfo(self.mano_id, i)[2] != pb.JOINT_FIXED]
        for idx in self.available_joints_indexes:
            pb.resetJointState(self.mano_id, idx, 0)
            pb.enableJointForceTorqueSensor(self.mano_id, idx, True)
        
        # add object and set object mass
        self.obj_mass = 0.2
        self.objId = self.addObject()


    def step(self, action):
        # set positoin can work @@
        # for i in range(len(self.available_joints_indexes)):
        #     pb.resetJointState(self.mano_id, self.available_joints_indexes[i], action[i])

        # pb.setJointMotorControlArray(bodyUniqueId=self.mano_id,
        #                             jointIndices=self.available_joints_indexes,
        #                             controlMode=pb.POSITION_CONTROL,
        #                             targetPositions=action,
        #                             forces=[100]*51)

        pb.stepSimulation()
        time.sleep(1/60)
        
        if self.cnt == 100:
            done = True
        else:
            self.cnt += 1
            done = False
        
        obs = self.getObservation()
        reward = self.getReward(obs)
        obs = np.zeros(57, dtype=np.float32)
        
        return obs, reward, done, {}


    def reset(self):
        self.cnt = 0

        # read training data
        with(open("./dgrasp_data.pickle", "rb")) as openfile:
            train_data = pickle.load(openfile)

        self.obj_init = train_data["subgoal_1"]["obj_init"]
        self.hand_traj_reach = train_data["subgoal_1"]["hand_traj_reach"]
        self.hand_traj_grasp = train_data["subgoal_1"]["hand_traj_grasp"]
        
        # self.obj_final = train_data["subgoal_1"]["obj_final"]
        # self.hand_ref_pose = np.squeeze(train_data["subgoal_1"]["hand_ref_pose"])[3:]
        # self.hand_ref_position = train_data["subgoal_1"]["hand_ref_position"]
        # self.hand_contact = train_data["subgoal_1"]["hand_contact"]

        # set initial object pose
        pb.resetBasePositionAndOrientation(self.objId, self.obj_init[:3], self.obj_init[3:])

        # set initial hand pose
        init_state = self.hand_traj_reach[0, 0]
        self.init_rot_ = Rot.from_euler('zxy', init_state[3:6]).as_matrix()
        self.init_or_ = np.transpose(self.init_rot_)

        for cnt, idx in enumerate(self.available_joints_indexes):
            pb.resetJointState(self.mano_id, idx, init_state[cnt])
        
        # set goals
        self.set_goals(train_data)

        # get observation
        obs = self.getObservation()
        obs = np.zeros(57, dtype=np.float32)

        return obs
    

    def set_goals(self, train_data):
        obj_goal_pos = train_data["subgoal_1"]["obj_final"]
        ee_goal_pos = train_data["subgoal_1"]["hand_ref_position"]
        goal_pose = np.squeeze(train_data["subgoal_1"]["hand_ref_pose"])[3:]
        goal_contacts = train_data["subgoal_1"]["hand_contact"]

        # set final object pose
        self.final_obj_pos_ = obj_goal_pos

        # convert object and handpose pose to rotation matrix format
        self.Obj_orientation_temp = Rot.from_quat(obj_goal_pos[3:]).as_matrix()
        quat_obj_init = obj_goal_pos[3:]
        Obj_orientation = np.transpose(self.Obj_orientation_temp)

        quat_goal_hand_w = Rot.from_euler('zxy', goal_pose[:3]).as_quat()
        root_pose_world_ = Rot.from_quat(quat_goal_hand_w).as_matrix()

        # Compute and set object relative goal hand pose
        quat_goal_hand_r = quatInvQuatMul(quat_obj_init, quat_goal_hand_w)
        rotm_goal_hand_r = Rot.from_quat(quat_goal_hand_r).as_matrix()
        euler_goal_pose = Rot.from_matrix(rotm_goal_hand_r).as_euler('zxy')

        self.final_pose_ = goal_pose
        self.final_pose_[:3] = euler_goal_pose

        # Compute and convert hand 3D joint positions into object relative frame
        tmp_rel_pos = ee_goal_pos - np.tile(self.final_obj_pos_[:3], (16, 1))
        self.final_ee_pos_ = Obj_orientation @ np.transpose(tmp_rel_pos)
        self.final_ee_pos_ = np.transpose(self.final_ee_pos_).reshape(48)

        # Intialize and set goal contact array
        self.num_active_contacts_ = np.sum(goal_contacts)
        self.final_contact_array_ = goal_contacts


    def render(self, mode='human'):
        view_matrix = pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = pb.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(960) /720,
                                                    nearVal=0.1,
                                                    farVal=100.0)
        pb.getCameraImage(width=960,
                        height=720,
                        viewMatrix=view_matrix,
                        projectionMatrix=proj_matrix,
                        renderer=pb.ER_BULLET_HARDWARE_OPENGL)


    def close(self):
        pb.disconnect()


    def getObservation(self):
        # 6D pose of the wrist(6) + joint angles(45)+ 6D pose of the object(6) + 
        # velocity of the wrist pose(6) + joint angular velocities(45) + velocity of the object pose(6)+
        # each joint forces exceeded on object(21)

        # hand tran(3) + hand rot(3) + hand pose(45) + obj tran(3) + obj rot(4)
        j = np.zeros(58, dtype=np.float32)
        j_vel = np.zeros(57, dtype=np.float32)
        
        # get hand pose and vel
        for i in range(len(self.available_joints_indexes)):
            joint_state = np.array(pb.getJointState(self.mano_id,self.available_joints_indexes[i]))[:2]
            j[i] = joint_state[0]
            j_vel[i] = joint_state[1]

        # get obj pose and vel
        obj_pose = pb.getBasePositionAndOrientation(self.objId)
        obj_pose = np.array(obj_pose[0] + obj_pose[1])

        obj_vel = pb.getBaseVelocity(self.objId)
        obj_vel = np.array(obj_vel[0] + obj_vel[1])
        j[51:] = obj_pose
        j_vel[51:] = obj_vel

        # get forces on links
        f = np.zeros(16)
        linkList = [6,  9,  12, 15,
                    18, 21, 24,
                    27, 30, 33,
                    36, 39, 42,
                    45, 48, 51]
        for cnt, l in enumerate(linkList):
            if len(pb.getContactPoints(bodyA=self.mano_id,bodyB=self.objId,linkIndexA=l)):
                contact=pb.getContactPoints(bodyA=self.mano_id,bodyB=self.objId,linkIndexA=l)
                force=np.zeros(3)
                for c in contact:
                    contact_normal=np.array(c[7])
                    contact_force=np.array(c[9])
                    force+=contact_force*contact_normal
                f[cnt]=np.linalg.norm(force)

        # feature extraction layers
        wrist_position = j[:3]
        wrist_orientation = Rot.from_euler('zyx', j[3:6]).as_matrix()
        wrist_orientation_transpose = np.transpose(wrist_orientation)

        obj_position = j[51:54]
        obj_orienttion = Rot.from_quat(j[54:]).as_matrix()

        # relative position between object and wrist in wrist coordinates
        rel_objpalm = wrist_position - obj_position
        rel_objpalm_pos = np.transpose(wrist_orientation) @ rel_objpalm

        # object displacement from initial position in wrist coordinates
        rel_obj_init = self.obj_init[:3] - obj_position
        rel_obj_pos_ = np.transpose(wrist_orientation) @ rel_obj_init

        # current global wirst pose in object relative frame
        rot_mult = obj_orienttion @ wrist_orientation
        euler_hand = Rot.from_matrix(rot_mult).as_euler('zyx')

        # difference between target and current global wrist pose
        self.rel_pose_ = self.final_pose_[:3] = euler_hand

        bodyLinearVel_ = j_vel[:3]
        bodyAngularVel_ = j_vel[3:6]

        rel_obj_vel = np.transpose(wrist_orientation) * j_vel[51:54] # relative object velocity
        rel_obj_qvel = np.transpose(wrist_orientation) * j_vel[54] # relative object angular velocity
        final_obj_pose_mat = Rot.from_quat(self.final_obj_pos_[3:]).as_matrix()

        final_obj_wrist = self.init_or_ @ final_obj_pose_mat # target object orientation in initial wrist frame
        obj_wrist = self.init_or_ @ self.Obj_orientation_temp # current object orientation in initial wrist frame
        obj_wrist_trans = np.transpose(obj_wrist)

        diff_obj_pose_mat = final_obj_wrist @ obj_wrist_trans # distance between current obj and target obj pose
        rel_obj_pose_r3 = Rot.from_matrix(diff_obj_pose_mat).as_euler('zxy') # convert to Euler
        rel_obj_pose_ = rel_obj_pose_r3

        '''
        '''

        obs = np.hstack((j, j_vel, f))
        return obs


    def getReward(self,obs):
        # 6D pose of the wrist(6) + joint angles(45)+ 6D pose of the object(6) + 
        # velocity of the wrist pose(6) + joint angular velocities(45) + velocity of the object pose(6)+
        # each joint forces exceeded on object(21)
        r = 0

        # rx=

        # rq = np.linalg.norm(obs[3:51] - self.hand_ref_pose)

        # have_force_mask = (obs[115:] > 0)
        # rc1 = np.dot(self.hand_contact, have_force_mask) / np.dot(self.hand_contact, self.hand_contact)
        # rc2 = min(np.dot(self.hand_contact, obs[115:]), 1 * self.obj_mass)

        # rreg1=

        return r


    def addObject(self):
        meshScale = np.array([1, 1, 1])
        visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                            fileName="./brick/type_D.obj",
                                            rgbaColor=[0, 0, 1, 1],
                                            meshScale=meshScale)
        collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                                fileName="./brick/type_D.obj",
                                                meshScale=meshScale)
        return pb.createMultiBody(
            self.obj_mass,
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )