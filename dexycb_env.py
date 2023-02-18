import os
import gym
import pickle
import pybullet as pb
import pybullet_data
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
import time

class DexYCBEnv(gym.Env): 
    def __init__(self, args, random_noise):
        # wrist translation(3) + wrist rotation(3) + joint rotation(45) = 51
        self.action_space = gym.spaces.box.Box(low=np.array([-1] * 3 + [-np.pi / 2] * 48, dtype=np.float32),
                                                high=np.array([1] * 3 + [np.pi / 2] * 48, dtype=np.float32))

        # wrist translation(3) + wrist rotation(3) + joint rotation(45) + lego translation(3) + lego rotation(3) = 57
        # joint angles(45) + joint angular velocities(45) + each joint forces exceeded on object(21) + 6D pose of the wrist(6) +
        # 6D pose of the object(6) + velocity of the wrist pose(6) + velocity of the object pose(6)
        self.observation_space = gym.spaces.box.Box(low=np.array([-100] * 279, dtype=np.float32),
                                                    high=np.array([100] * 279, dtype=np.float32))

        # connect to pybullet
        if args.GUI:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)

        pb.setTimeStep(1 / 60)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)

        # set plane and table
        # pb.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.plane_id = pb.loadURDF("plane.urdf")

        # add mano urdf
        self.mano_id = pb.loadURDF("/manoUrdf/20200709-subject-01_right/mano_addTips.urdf", [0, 0, 0], pb.getQuaternionFromEuler([0, 0, 0]))
        self.available_joints_indexes = [i for i in range(pb.getNumJoints(self.mano_id)) if pb.getJointInfo(self.mano_id, i)[2] != pb.JOINT_FIXED]
        
        # reset hand pose and enable forcetorquesensor
        for idx in self.available_joints_indexes:
            pb.resetJointState(self.mano_id, idx, 0)
            pb.enableJointForceTorqueSensor(self.mano_id, idx, True)
        
        # set joint limit
        self.joint_limit_low, self.joint_limit_high = np.zeros(51), np.zeros(51)
        
        for cnt, idx in enumerate(self.available_joints_indexes[6:], start=6):
            tmp_low, tmp_high = pb.getJointInfo(self.mano_id, idx)[8:10]
            self.joint_limit_low[cnt] = tmp_low
            self.joint_limit_high[cnt] = tmp_high
        
        self.joint_limit_low[:3].fill(-2)
        self.joint_limit_low[3:6].fill(-np.pi)

        self.joint_limit_high[:3].fill(2)
        self.joint_limit_high[3:6].fill(np.pi)

        # initialize actionMean_
        self.actionMean_ = np.zeros(51)
        
        for i in range(51):
            self.actionMean_[i] = (self.joint_limit_low[i] + self.joint_limit_high[i]) / 2

        # set action scaling
        self.actionStd_ = np.full(51, 0.015)
        self.actionStd_[:3].fill(0.001)
        self.actionStd_[3:6].fill(0.01)

        # initialize 3D positions weights for fingertips higher than for other fingerparts
        self.finger_weights_ = np.full(63, 1, dtype=np.float32)
        self.finger_weights_[12:15] *= 4
        self.finger_weights_[24:27] *= 4
        self.finger_weights_[36:39] *= 4
        self.finger_weights_[48:51] *= 4
        self.finger_weights_[60:63] *= 4
        self.finger_weights_ /= np.sum(self.finger_weights_)
        self.finger_weights_ *= 63

        # create link to joint list
        self.linkToJointList = [6,9,12,15,16,
                                19,22,25,26,
                                29,32,35,36,
                                39,42,45,46,
                                49,52,55,56]
        self.linkList = [6,9,12,15,
                        19,22,25,
                        29,32,35,
                        39,42,45,
                        49,52,55]
        
        # add link friction
        # for l in self.linkToJointList:
        #     pb.changeDynamics(bodyUniqueId=self.mano_id,linkIndex=l,lateralFriction=2)

        # add motion_synthesis flag
        self.motion_synthesis = False
        
        # add object and set object mass
        self.obj_mass = 0.1 # 0.05
        self.objId = self.addObject()

        # arguments
        self.args = args
        self.random_noise_pos = random_noise[0]
        self.random_noise_qpos = random_noise[1]

        # training data
        self.data_id = 0
        with open("./dexycb_data_all.pickle", "rb") as openfile:
            self.train_data = pickle.load(openfile)


    def step(self, action):
        if True: # use predict result
            if self.motion_synthesis:
                mp_final_world = np.copy(self.cur_train_data["subgoal_1"]["hand_traj_grasp"][-1, 0].reshape(51)[:3])
                act_pos = mp_final_world - self.init_state[:3]
                act_or_pose = self.init_or_ @ act_pos
            else:
                # Convert final root hand translation back from (current) object into world frame
                obj_pose = pb.getBasePositionAndOrientation(self.objId)
                obj_pose = np.array(obj_pose[0] + obj_pose[1])
                Obj_Position = obj_pose[:3]
                Obj_orientation_temp = Rot.from_quat(obj_pose[3:]).as_matrix()

                Fpos_world = Obj_orientation_temp @ self.final_mp_base
                Fpos_world += Obj_Position

                act_pos = Fpos_world - self.init_state[:3] # compute distance of curent root to initial root in world frame
                act_or_pose = self.init_or_ @ act_pos # rotate the world coordinate into hand's origin frame (from the start of the episode)
            
            self.actionMean_[:3] = act_or_pose

            # Compute position target for actuators
            action = action * self.actionStd_ # residual action * scaling
            action += self.actionMean_ # add wrist bias (first 3DOF) and last pose (48DoF)

            # clipped the action
            action_clipped = np.minimum(np.maximum(action, self.joint_limit_low), self.joint_limit_high)

            pb.setJointMotorControlArray(bodyUniqueId=self.mano_id,
                                        jointIndices=self.available_joints_indexes,
                                        controlMode=pb.POSITION_CONTROL,
                                        targetPositions=action_clipped,
                                        forces=[5] * 51)
        else:
            pb.setJointMotorControlArray(bodyUniqueId=self.mano_id,
                                        jointIndices=self.available_joints_indexes,
                                        controlMode=pb.POSITION_CONTROL,
                                        targetPositions=action,
                                        forces=[5] * 51)

        pb.stepSimulation()
        time.sleep(1/60)
        
        obs = self.getObservation()
        
        # update actionMean_
        for i in range(len(self.available_joints_indexes)):
            self.actionMean_[i] = pb.getJointState(self.mano_id,self.available_joints_indexes[i])[0]

        reward = self.getReward()
        
        return obs, reward, False, {}


    def reset(self):
        self.step_cnt = 0
        # self.data_id = (self.data_id + 1) % 21
        self.data_id = np.random.randint(21)
        self.cur_train_data = self.train_data[self.data_id]

        self.obj_init = np.copy(self.cur_train_data["subgoal_1"]["obj_init"])
        self.hand_traj_reach = np.copy(self.cur_train_data["subgoal_1"]["hand_traj_reach"])
        self.hand_traj_grasp = np.copy(self.cur_train_data["subgoal_1"]["hand_traj_grasp"])

        # set initial object pose
        pb.resetBasePositionAndOrientation(self.objId, self.obj_init[:3], self.obj_init[3:])

        # set initial hand pose
        self.init_state = self.hand_traj_reach[0, 0].copy() # make init hand pose near the object

        # add random noise
        self.init_state[:3] += self.random_noise_pos[:3]
        self.init_state[3:] += self.random_noise_qpos
        
        self.init_rot_ = Rot.from_euler('XYZ', self.init_state[3:6]).as_matrix()
        self.init_or_ = np.transpose(self.init_rot_)

        # set mano urdf base position and orientation
        pb.resetBasePositionAndOrientation(self.mano_id, self.init_state[:3], Rot.from_euler('XYZ', self.init_state[3:6]).as_quat())

        # set joint values
        for cnt, idx in enumerate(self.available_joints_indexes):
            if cnt > 5:
                pb.resetJointState(self.mano_id, idx, self.init_state[cnt])
                self.actionMean_[cnt] = self.init_state[cnt]
            else:
                pb.resetJointState(self.mano_id, idx, 0)
                self.actionMean_[cnt] = 0
        
        # set goals
        self.set_goals()
        
        # init motion_synthesis flag
        self.motion_synthesis = False

        # get observation
        obs = self.getObservation()

        return obs
    

    def set_goals(self):
        obj_goal_pos = np.copy(self.cur_train_data["subgoal_1"]["obj_final"]) # not same as original code in dgrasp !!!!!!!!!!!!!!
        ee_goal_pos = np.copy(self.cur_train_data["subgoal_1"]["hand_ref_position"])
        goal_pose = np.copy(self.cur_train_data["subgoal_1"]["hand_ref_pose"].reshape(51)[3:])
        goal_contacts = np.copy(self.cur_train_data["subgoal_1"]["hand_contact"])

        # set final hand pose in world frame
        self.final_pose_world = np.copy(self.cur_train_data["subgoal_1"]["hand_ref_pose"].reshape(51))

        # set final object pose
        self.final_obj_pos_ = np.copy(obj_goal_pos)

        # convert object and handpose pose to rotation matrix format
        Obj_orientation_temp = Rot.from_quat(obj_goal_pos[3:]).as_matrix()
        # quat_obj_init = np.copy(obj_goal_pos[3:])
        Obj_orientation = np.transpose(Obj_orientation_temp)

        quat_goal_hand_w = Rot.from_euler('XYZ', goal_pose[:3]).as_quat()
        root_pose_world_ = Rot.from_quat(quat_goal_hand_w).as_matrix()

        # Compute and set object relative goal hand pose
        # quat_goal_hand_r = quatInvQuatMul(quat_obj_init, quat_goal_hand_w)
        # rotm_goal_hand_r = Rot.from_quat(quat_goal_hand_r).as_matrix()
        rotm_goal_hand_r = Obj_orientation @ root_pose_world_
        euler_goal_pose = Rot.from_matrix(rotm_goal_hand_r).as_euler('XYZ')
        self.final_pose_ = np.copy(goal_pose)
        self.final_pose_[:3] = euler_goal_pose

        # Compute and convert hand 3D joint positions into object relative frame
        tmp_rel_pos = ee_goal_pos - np.tile(self.final_obj_pos_[:3], (21, 1))
        self.final_ee_pos_ = Obj_orientation @ np.transpose(tmp_rel_pos)
        self.final_ee_pos_ = np.transpose(self.final_ee_pos_)

        # convert mano pybullet hand base translation into object relative frame
        tmp_rel_pos = self.cur_train_data["subgoal_1"]["hand_ref_pose"].reshape(51)[:3] - self.obj_init[:3]
        self.final_mp_base = Obj_orientation @ np.transpose(tmp_rel_pos)

        # Intialize and set goal contact array
        num_active_contacts_ = np.sum(goal_contacts)
        self.final_contact_array_ = np.copy(goal_contacts)
        self.k_contact = 1.0 / num_active_contacts_


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
            joint_state = np.array(pb.getJointState(self.mano_id,self.available_joints_indexes[i])[:2])
            j[i] = joint_state[0]
            j_vel[i] = joint_state[1]
        
        # get hand joint position
        position_list = []
        for cnt, joint in enumerate(self.linkToJointList):
            linkState = pb.getLinkState(self.mano_id, joint)
            position = np.array(linkState[0])
            position_list.append(position)
        j_pos = np.array(position_list, dtype=np.float32).reshape(63)

        # get obj pose and vel
        obj_pose = pb.getBasePositionAndOrientation(self.objId)
        obj_pose = np.array(obj_pose[0] + obj_pose[1])

        obj_vel = pb.getBaseVelocity(self.objId)
        obj_vel = np.array(obj_vel[0] + obj_vel[1])
        j[51:] = obj_pose
        j_vel[51:] = obj_vel

        # feature extraction layers
        wrist_position = np.copy(j_pos[:3])
        wrist_orientation = Rot.from_euler('XYZ', j[3:6]).as_matrix()
        wrist_orientation_transpose = np.transpose(wrist_orientation)

        obj_position = np.copy(j[51:54])
        Obj_orientation_temp = Rot.from_quat(j[54:]).as_matrix()
        Obj_orientation = np.transpose(Obj_orientation_temp)

        # compute relative hand pose
        self.rel_pose_ = self.final_pose_ - j[3:51]

        # compute object pose in wrist frame
        palm_world_pose_mat = np.copy(wrist_orientation)
        palm_world_pose_mat_trans = np.transpose(palm_world_pose_mat)
        obj_pose_wrist_mat = palm_world_pose_mat_trans @ Obj_orientation_temp
        obj_pose_ = Rot.from_matrix(obj_pose_wrist_mat).as_euler('XYZ')

        # relative position between object and wrist in wrist coordinates
        rel_objpalm = wrist_position - obj_position
        rel_objpalm_pos = np.transpose(wrist_orientation) @ rel_objpalm

        # z-distance to the table in wrist coordinates
        rel_body_table_pos_ = np.transpose(wrist_orientation) @ np.array([0, 0, j_pos[2]])

        # object displacement from initial position in wrist coordinates
        rel_obj_init = self.obj_init[:3] - obj_position
        self.rel_obj_pos_ = np.transpose(wrist_orientation) @ rel_obj_init

        # current global wirst pose in object relative frame
        rot_mult = Obj_orientation @ wrist_orientation
        euler_hand = Rot.from_matrix(rot_mult).as_euler('XYZ')

        # difference between target and current global wrist pose
        self.rel_pose_[:3] = self.final_pose_[:3] - euler_hand

        self.bodyLinearVel_ = j_vel[:3]
        self.bodyAngularVel_ = j_vel[3:6]

        self.rel_obj_vel = np.transpose(np.transpose(wrist_orientation) @ j_vel[51:54]) # relative object velocity
        rel_obj_qvel = np.transpose(np.transpose(wrist_orientation) @ j_vel[54:]) # relative object angular velocity
        final_obj_pose_mat = Rot.from_quat(self.final_obj_pos_[3:]).as_matrix()

        final_obj_wrist = self.init_or_ @ final_obj_pose_mat # target object orientation in initial wrist frame
        obj_wrist = self.init_or_ @ Obj_orientation_temp # current object orientation in initial wrist frame
        obj_wrist_trans = np.transpose(obj_wrist)

        diff_obj_pose_mat = final_obj_wrist @ obj_wrist_trans # distance between current obj and target obj pose
        rel_obj_pose_r3 = Rot.from_matrix(diff_obj_pose_mat).as_euler('XYZ') # convert to Euler
        rel_obj_pose_ = rel_obj_pose_r3

        # Compute relative 3D position features for all hand joints
        tmp_rel_pos = j_pos.reshape(21, 3) - np.tile(obj_position, (21, 1))
        Rel_fpos = Obj_orientation @ np.transpose(tmp_rel_pos)
        Rel_fpos = np.transpose(Rel_fpos) # compute current relative pose in object coordinates

        obj_frame_diff = self.final_ee_pos_ - Rel_fpos # distance between target 3D positions and current 3D positions in object frame
        obj_frame_diff_w = Obj_orientation_temp @ np.transpose(obj_frame_diff) # convert distances to world frame
        obj_frame_diff_w = np.transpose(obj_frame_diff_w)

        obj_frame_diff_h = wrist_orientation_transpose @ np.transpose(obj_frame_diff_w)
        obj_frame_diff_h = np.transpose(obj_frame_diff_h) # convert distances to wrist frame
        self.rel_body_pos_ = obj_frame_diff_h.reshape(63)

        # compute current contacts of hand parts and the contact force
        self.z_impulse = 0
        self.impulses_ = np.zeros(16)
        for cnt, l in enumerate(self.linkList):
            if len(pb.getContactPoints(bodyA=self.mano_id, bodyB=self.objId, linkIndexA=l)):
                contact = pb.getContactPoints(bodyA=self.mano_id, bodyB=self.objId, linkIndexA=l)
                force = np.zeros(3)
                for c in contact:
                    contact_normal = np.array(c[7])
                    contact_force = np.array(c[9])
                    force += (contact_force * contact_normal)
                # self.z_impulse += force[2]
                # if force[2] > 0:
                #     force.fill(0)
                self.impulses_[cnt] = np.linalg.norm(force)
        
        # compute relative target contact vector, i.e., which goal contacts are currently in contact
        self.rel_contacts_ = self.final_contact_array_ * (self.impulses_ > 0)

        # add all features to observation
        obs = np.hstack([j[:51],
                        self.bodyLinearVel_,
                        self.bodyAngularVel_,
                        j_vel[6:51],
                        self.rel_body_pos_,
                        self.rel_pose_,
                        rel_objpalm_pos,
                        self.rel_obj_vel,
                        rel_obj_qvel,
                        self.final_contact_array_,
                        self.impulses_,
                        self.rel_contacts_,
                        self.rel_obj_pos_,
                        rel_body_table_pos_,
                        obj_pose_]).astype(np.float32)
        
        return obs


    def getReward(self):
        # 6D pose of the wrist(6) + joint angles(45)+ 6D pose of the object(6) + 
        # velocity of the wrist pose(6) + joint angular velocities(45) + velocity of the object pose(6)+
        # each joint forces exceeded on object(21)
        
        # Compute general reward terms
        pose_reward_ = -np.linalg.norm(self.rel_pose_)
        pos_reward_ = -np.linalg.norm(self.rel_body_pos_ * self.finger_weights_) ** 2 # without finger_weights

        # Compute regularization rewards
        rel_obj_reward_ = np.linalg.norm(self.rel_obj_vel) ** 2
        body_vel_reward_ = np.linalg.norm(self.bodyLinearVel_) ** 2
        body_qvel_reward_ = np.linalg.norm(self.bodyAngularVel_) ** 2
        contact_reward_ = self.k_contact * np.sum(self.rel_contacts_)
        impulse_reward_ = np.sum(self.final_contact_array_ * self.impulses_)

        # Store all rewards
        rewards = 0
        rewards += 2.0 * max(-10.0, pos_reward_) # pos_reward
        rewards += 0.1 * max(-10.0, pose_reward_) # pose_reward
        rewards += 1.0 * max(-10.0, contact_reward_) # contact_reward
        rewards += 2.0 * min(impulse_reward_, self.obj_mass) # impulse_reward
        rewards += -1.0 * max(0.0, rel_obj_reward_) # rel_obj_reward_
        rewards += -0.5 * max(0.0, body_vel_reward_) # body_vel_reward_
        rewards += -0.5 * max(0.0, body_qvel_reward_) # body_qvel_reward_
        # rewards += -1 * min(self.obj_mass,self.z_impulse)

        # print("pos_reward:",2.0 * max(-10.0, pos_reward_))
        # print("pose_reward:",0.2 * max(-10.0, pose_reward_))
        # print("contact_reward:",0.2 * max(-10.0, contact_reward_))
        # print("impulse_reward:",1.0 * min(impulse_reward_, self.obj_mass))
        # print("rel_obj_reward_:",-1.0 * max(0.0, rel_obj_reward_))
        # print("body_vel_reward_:",-0.5 * max(0.0,body_vel_reward_))
        # print("body_qvel_reward_:",-0.5 * max(0.0,body_qvel_reward_))
        # print("z_impulse:",-1* min(self.obj_mass,self.z_impulse))
        # print("-"*10)

        return rewards


    def set_root_control(self):
        self.motion_synthesis = True


    def addObject(self):
        meshScale = np.array([1, 1, 1])
        visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                            fileName="/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb-20210415/models/002_master_chef_can/textured_simple.obj",
                                            rgbaColor=[0, 0, 1],
                                            meshScale=meshScale)
        collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                                fileName="/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb-20210415/models/002_master_chef_can/textured_simple.obj",
                                                meshScale=meshScale)
        meshId = pb.createMultiBody(
            self.obj_mass,
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )

        return meshId