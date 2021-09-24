from shutil import rmtree
import copy
import csv
import math
import os
import pickle
import random
import time

import cv2
import numpy as np
import pybullet as pb
import pyro
import pyro.distributions as dist
import torch
import transforms3d as tf3d
from PIL import Image
from tqdm import trange

import generation.calibrations as calibrations
from generation.mujocoCabinetParts import build_cabinet, sample_cabinet
from generation.mujocoDoubleCabinetParts import (build_cabinet2,
                                                 sample_cabinet2,
                                                 set_two_door_control)
from generation.mujocoDrawerParts import build_drawer, sample_drawers
from generation.mujocoMicrowaveParts import build_microwave, sample_microwave
from generation.mujocoRefrigeratorParts import (build_refrigerator,
                                                sample_refrigerator)
from generation.mujocoToasterOvenParts import build_toaster, sample_toaster
from generation.utils import *

pb_client = pb.connect(pb.GUI)
# pb.setGravity(0, 0, -100)
pb.setRealTimeSimulation(True)


def white_bg(img):
    mask = img == 0
    img_cp = copy.deepcopy(img)
    img_cp[mask.all(axis=2)] = [255, 255, 255, 0]
    return img_cp


def buffer_to_real(d, d_far, d_near):
    return d_far * d_near / (d_far - (d_far - d_near) * d)


def vertical_flip(img):
    return np.flip(img, axis=0)


class SceneGenerator():
    def __init__(self, root_dir='bull/test_cabinets/solo', masked=False, debug_flag=False):
        '''
        Class for generating simulated articulated object dataset.
        params:
            - root_dir: save in this directory
            - start_idx: index of first image saved - useful in threading context
            - depth_data: np array of depth images
            - masked: should the background of depth images be 0s or 1s?
        '''
        self.scenes = []
        self.root_dir = root_dir
        self.masked = masked
        self.depth_data = []
        self.debugging = debug_flag

        # TODO: these seem to be always overrided
        # Camera external settings
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[4, 0, 1],
            cameraTargetPosition=[0, 0, 1],
            cameraUpVector=[0, 0, 1]
        )

        self.d_near = 0.1
        self.d_far = 8.1

        # Camera internal settings
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.,
            aspect=1.0,
            nearVal=self.d_near,
            farVal=self.d_far
        )

    def write_urdf(self, filename, xml):
        with open(filename, "w") as text_file:
            text_file.write(xml)

    def sample_obj(self, obj_type, mean_flag, left_only, cute_flag=False):
        if obj_type == 'microwave':
            l, w, h, t, left, _ = sample_microwave(mean_flag)
            if mean_flag:
                obj = build_microwave(l, w, h, t, left,
                                      set_pos=[0.0, 0.0, 0.0],
                                      set_rot=[1.0, 0.0, 0.0, 0.0])
            elif cute_flag:
                _, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_microwave(l, w, h, t, left,
                                      set_pos=[0.0, 0.0, 0.0],
                                      set_rot=base_quat)
            else:
                obj = build_microwave(l, w, h, t, left)

            camera_dist = 2
            camera_height = h

        elif obj_type == 'drawer':
            l, w, h, t, left, mass = sample_drawers(mean_flag)
            if mean_flag:
                obj = build_drawer(l, w, h, t, left,
                                   set_pos=[1.5, 0.0, -0.4],
                                   set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_drawer(l, w, h, t, left,
                                   set_pos=[1.2, 0.0, -0.15],
                                   set_rot=base_quat)
            else:
                obj = build_drawer(l, w, h, t, left)

            camera_dist = max(2, 2 * math.log(10 * h))
            camera_height = h / 2.

        elif obj_type == 'toaster':
            l, w, h, t, left, mass = sample_toaster(mean_flag)
            if mean_flag:
                obj = build_toaster(l, w, h, t, left,
                                    set_pos=[1.5, 0.0, -0.3],
                                    set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_toaster(l, w, h, t, left,
                                    set_pos=[1.0, 0.0, -0.15],
                                    set_rot=base_quat)
            else:
                obj = build_toaster(l, w, h, t, left)

            camera_dist = max(1, 2 * math.log(10 * h))
            camera_height = h / 2.

        elif obj_type == 'cabinet':
            l, w, h, t, left, mass = sample_cabinet(mean_flag)
            if mean_flag:
                if left_only:
                    left = True
                else:
                    left = False
                obj = build_cabinet(l, w, h, t, left,
                                    set_pos=[1.5, 0.0, -0.3],
                                    set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_cabinet(l, w, h, t, left,
                                    set_pos=[1.5, 0.0, -0.15],
                                    set_rot=base_quat)
            else:
                left = np.random.choice([True, False])
                obj = build_cabinet(l, w, h, t, left)

            camera_dist = 2 * math.log(10 * h)
            camera_height = h / 2.

        elif obj_type == 'cabinet2':
            l, w, h, t, left, mass = sample_cabinet2(mean_flag)
            if mean_flag:
                obj = build_cabinet2(l, w, h, t, left,
                                     set_pos=[1.5, 0.0, -0.3],
                                     set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_cabinet2(l, w, h, t, left,
                                     set_pos=[1.5, 0.0, -0.15],
                                     set_rot=base_quat)
            else:
                obj = build_cabinet2(l, w, h, t, left)

            camera_dist = 2 * math.log(10 * h)
            camera_height = h / 2.

        elif obj_type == 'refrigerator':
            l, w, h, t, left, mass = sample_refrigerator(mean_flag)
            if mean_flag:

                obj = build_refrigerator(l, w, h, t, left,
                                         set_pos=[1.5, 0.0, -0.3],
                                         set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_refrigerator(l, w, h, t, left,
                                         set_pos=[2.5, 0.0, -0.75],
                                         set_rot=base_quat)

            else:
                obj = build_refrigerator(l, w, h, t, left)

            camera_dist = 2 * math.log(10 * h)
            camera_height = h / 2.

        else:
            raise 'uh oh, object not implemented!'
        return obj, camera_dist, camera_height

    def generate_scenes(self, N, obj_type, mean_flag=False, left_only=False, cute_flag=False, test=False, video=False):
        self.save_dir = os.path.join(self.root_dir, obj_type)
        if test:
            self.save_dir += '-test'
        if os.path.isdir(self.save_dir):
            rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        print('Generating data in %s...' % self.save_dir)
        fname = os.path.join(self.save_dir, 'labels.csv')
        with open(fname, 'w') as f:
            writ = csv.writer(f, delimiter=',')
            writ.writerow(['Object Name', 'Joint Type', 'Image Index', 'l_1', 'l_2', 'l_3', 'm_1', 'm_2', 'm_3'])
            for i in trange(N):
                obj, camera_dist, camera_height = self.sample_obj(obj_type, mean_flag, left_only, cute_flag=cute_flag)
                fname = os.path.join(self.save_dir, 'scene' + str(i).zfill(6) + '.xml')
                self.write_urdf(fname, obj.xml)
                self.scenes.append(fname)
                self.take_images(fname, obj, camera_dist, camera_height, obj.joint_index, writ, test=test, video=video)

    def take_images(self, filename, obj, camera_dist, camera_height, joint_index, writer, img_idx=0, debug=False, test=False, video=False):
        # TODO: img_idx should be tracked across different samples so that they won't overwrite previous ones
        obj_id, _ = pb.loadMJCF(filename)

        # create normal texture image
        # x, y = np.meshgrid(np.linspace(-1, 1, 1280), np.linspace(-1, 1, 1280))
        # x, y = np.meshgrid(np.linspace(-1, 1, 300), np.linspace(-1, 1, 300))
        # texture_img = (72 * (np.stack([np.cos(30 * x), np.cos(30 * y), np.cos(30 * (x + y))]) + 2)).astype(np.uint8).transpose(1, 2, 0)
        # texture_img = Image.fromarray(texture_img)
        # fname = 'normal_texture.png'
        # texture_img.save(fname)
        # textureId = pb.loadTexture(fname, physicsClientId=pb_client)

        # create gaussian noise texture image
        SHAPE = (300, 300)
        noise = np.random.normal(255, 180, SHAPE)
        image_noise = Image.fromarray(noise)
        image_noise = image_noise.convert('RGB')
        fname = "gaussian_noise.png"
        image_noise.save(fname)
        texture_id = pb.loadTexture(fname, physicsClientId=pb_client)

        n_joints = pb.getNumJoints(obj_id)
        colors = np.random.random_sample((n_joints + 1, 3))
        colors = np.hstack((colors, np.ones((n_joints + 1, 1))))
        for idx in range(-1, n_joints):
            if idx == joint_index:
                continue
            color = colors[idx + 1]
            pb.changeVisualShape(obj_id, idx, textureUniqueId=texture_id, rgbaColor=color, specularColor=color, physicsClientId=pb_client)

        # theta = np.random.uniform(low=-np.pi / 3, high=np.pi / 3)
        # cameraEyePosition = np.array([camera_dist * np.cos(theta), camera_dist * np.sin(theta), 1])
        cameraEyePosition = np.array([camera_dist, 0, camera_height + 1])
        cameraUpVector = np.array([0, 0, 1])
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=cameraEyePosition,
            cameraTargetPosition=[0, 0, camera_height],
            cameraUpVector=cameraUpVector
        )

        # startPos = [0, 0, 0]
        # state = {'startPos': startPos, 'eulerOrientation': [], 'joints': [], 'joint_index': joint_index, 'img_idx': img_idx}
        # obj_rotation = np.random.uniform(-np.pi/4.,np.pi/4.)
        # obj_rotation = 0
        # startOrientation = pb.getQuaternionFromEuler([0, 0, obj_rotation])
        # state['eulerOrientation'] = [0, 0, obj_rotation]

        # pb.resetBasePositionAndOrientation(obj_id, startPos, startOrientation)
        pb.setJointMotorControl2(obj_id, joint_index, controlMode=pb.VELOCITY_CONTROL, targetVelocity=-1, force=500)
        # Take 16 pictures, permuting orientation and joint extension
        for t in range(32):
            #########################
            IMG_WIDTH = calibrations.sim_width
            IMG_HEIGHT = calibrations.sim_height
            #########################

            # # Take picture without texture
            # width, height, img, depth, segImg = pb.getCameraImage(
            #     IMG_WIDTH, # width
            #     IMG_HEIGHT, # height
            #     self.viewMatrix,
            #     self.projectionMatrix,
            #     lightDirection=[camera_dist, 0, camera_height+1], # light source
            #     shadow=1, # include shadows
            # )

            # use projective texture, it's more robust, applies texture on all sides at once
            # viewMat = [
            #     0.642787516117096, -0.4393851161003113, 0.6275069713592529, 0.0, 0.766044557094574,
            #     0.36868777871131897, -0.5265407562255859, 0.0, -0.0, 0.8191521167755127, 0.5735764503479004,
            #     0.0, 2.384185791015625e-07, 2.384185791015625e-07, -5.000000476837158, 1.0
            # ]
            # projMat = [
            #     0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0,
            #     0.0, 0.0, -0.02000020071864128, 0.0
            # ]

            # cam = pb.getDebugVisualizerCamera()
            # viewMat = cam[2]
            # projMat = cam[3]

            width, height, img, depth, _ = pb.getCameraImage(
                IMG_WIDTH,  # width
                IMG_HEIGHT,  # height
                self.viewMatrix,
                self.projectionMatrix,
                # renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                # renderer=pb.ER_TINY_RENDERER,
                # flags=pb.ER_USE_PROJECTIVE_TEXTURE,
                # projectiveTextureView=viewMat,
                # projectiveTextureProj=projMat
            )

            img = np.asarray(img).reshape((height, width, 4))
            depth = np.asarray(depth).reshape((height, width))

            # if test:
            #     state['viewMatrix'] = self.viewMatrix
            #     state['projectionMatrix'] = self.projectionMatrix
            #     # state['lightDirection'] = [camera_dist, 0, camera_height + 1]
            #     state['height'] = IMG_HEIGHT
            #     state['width'] = IMG_WIDTH
            #     state['mjcf'] = filename

            #     config_name = os.path.join(self.save_dir, 'config' + str(img_idx).zfill(6) + '.pkl')
            #     f = open(config_name, "wb")
            #     pickle.dump(state, f)
            #     f.close()

            #depth = vertical_flip(depth)
            real_depth = buffer_to_real(depth, self.d_far, self.d_near)
            norm_depth = real_depth / self.d_far

            if self.masked:
                # set background depth to 0
                mask = norm_depth < 0.99
                norm_depth *= mask

            if self.debugging:
                # save image to disk for visualization
                # img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))

                #img = vertical_flip(img)

                # img = white_bg(img)  # TODO
                img_fname = os.path.join(self.save_dir, 'img' + str(img_idx).zfill(6) + '.png')
                depth_fname = os.path.join(self.save_dir, 'depth' + str(img_idx).zfill(6) + '.png')
                integer_depth = norm_depth * 255
                cv2.imwrite(img_fname, img)
                cv2.imwrite(depth_fname, integer_depth)

            # if IMG_WIDTH != 192 or IMG_HEIGHT != 108:
            #     depth = cv2.resize(norm_depth, (192,108))

            if joint_index is None:
                raise Exception("Joint index not defined! Are you simulating a 2DOF object? (Don't do that yet)")

            large_door_joint_info = pb.getJointInfo(obj_id, joint_index)
            p = np.array(list(large_door_joint_info[14]))
            l = np.array(list(large_door_joint_info[13]))
            m = np.cross(large_door_joint_info[14], large_door_joint_info[13])

            depth_fname = os.path.join(self.save_dir, 'depth' + str(img_idx).zfill(6) + '.pt')
            torch.save(norm_depth, depth_fname)
            row = np.concatenate((np.array([obj.name, obj.joint_type, img_idx]), l, m))  # SAVE SCREW REPRESENTATION HERE
            writer.writerow(row)

            # if video:
            #     increments = {j: 0 for j in range(pb.getNumJoints(obj_id))}
            #     videoFolderFname = os.path.join(self.save_dir, 'video_for_img_' + str(img_idx).zfill(6))
            #     os.makedirs(videoFolderFname, exist_ok=False)
            #     for frame_idx in range(90):
            #         for j in range(pb.getNumJoints(obj_id)):
            #             pb.resetJointState(obj_id, j, increments[j])
            #             increments[j] += obj.control[j] / 90

            #         _, _, rgbFrame, depthFrame, _ = pb.getCameraImage(
            #             IMG_WIDTH,  # width
            #             IMG_HEIGHT,  # height
            #             self.viewMatrix,
            #             self.projectionMatrix,
            #             lightDirection=[camera_dist, 0, camera_height + 1],  # light source
            #             shadow=1,  # include shadows
            #         )

            #         frameRgbFname = os.path.join(videoFolderFname, 'rgb_frame_' + str(frame_idx).zfill(6) + '.png')
            #         rgbFrame = white_bg(rgbFrame)
            #         cv2.imwrite(frameRgbFname, rgbFrame)
            #         frameDepthFname = os.path.join(videoFolderFname, 'depth_frame_' + str(frame_idx).zfill(6) + '.pt')
            #         real_depth = buffer_to_real(depthFrame, 12.0, 0.1)
            #         norm_depth = real_depth / 12.0

            #         if self.masked:
            #             # remove background
            #             mask = norm_depth > 0.99
            #             norm_depth = (1 - mask) * norm_depth

            #         torch.save(torch.tensor(norm_depth.copy()), frameDepthFname)

            img_idx += 1
            # time.sleep(0.05)
        pb.removeBody(obj_id)
