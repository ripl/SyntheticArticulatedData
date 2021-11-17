import csv
import math
import os
from shutil import rmtree

import ffmpeg
import numpy as np
import pybullet as pb
import torch
import transforms3d as tf3d
from PIL import Image
from tqdm import trange

import generation.calibrations as calibrations
from generation.mujocoCabinetParts import build_cabinet, sample_cabinet
from generation.mujocoDoubleCabinetParts import build_cabinet2, sample_cabinet2
from generation.mujocoDrawerParts import build_drawer, sample_drawers
from generation.mujocoMicrowaveParts import build_microwave, sample_microwave
from generation.mujocoRefrigeratorParts import (build_refrigerator,
                                                sample_refrigerator)
from generation.mujocoToasterOvenParts import build_toaster, sample_toaster
from generation.utils import *

pb_client = pb.connect(pb.DIRECT)
pb.setRealTimeSimulation(True)


def buffer_to_real(d, d_far, d_near):
    return d_far * d_near / (d_far - (d_far - d_near) * d)


def proj2intrinsics(proj, w, h):
    K = np.zeros((3, 3))
    K[0, 0] = proj[0] * w / 2
    K[0, 2] = w / 2
    K[1, 1] = proj[5] * h / 2
    K[1, 2] = h / 2
    K[2, 2] = 1
    return K


def view2extrinsics(view):
    return np.array([[view[0], view[4], view[8], view[12]],
                     [-view[1], -view[5], -view[9], -view[13]],
                     [-view[2], -view[6], -view[10], -view[14]]])


class SceneGenerator():
    def __init__(self, root_dir, mode, masked=False, debug_flag=False):
        '''
        Class for generating simulated articulated object dataset.
        params:
            - root_dir: save in this directory
            - start_idx: index of first image saved - useful in threading context
            - masked: should the background of depth images be 0s or 1s?
        '''
        self.scenes = []
        self.root_dir = root_dir
        self.mode = mode
        self.masked = masked
        self.debugging = debug_flag

        self.d_near = 0.1
        self.d_far = 8.1

        # Camera internal settings
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.,
            aspect=1.0,
            nearVal=self.d_near,
            farVal=self.d_far
        )
        self.K = proj2intrinsics(self.projectionMatrix, calibrations.sim_width, calibrations.sim_height)

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

            camera_dist = 2.5
            target_height = h

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
            target_height = h / 2.

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
            target_height = h / 2.

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
            target_height = h / 2.

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
            target_height = h / 2.

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
            target_height = h / 2.

        else:
            raise 'uh oh, object not implemented!'
        return obj, camera_dist, target_height

    def generate_scenes(self, N, obj_type, mean_flag=False, left_only=False, cute_flag=False, test=False):
        self.save_dir = os.path.join(self.root_dir, obj_type)
        if test:
            self.save_dir += '-test'
        if os.path.isdir(self.save_dir):
            rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        print('Generating data in %s...' % self.save_dir)
        np.save(os.path.join(self.save_dir, 'cam_intrinsics.npy'), self.K)
        fname = os.path.join(self.save_dir, 'labels.csv')
        with open(fname, 'w') as f:
            writ = csv.writer(f, delimiter=',')
            writ.writerow(['Object', 'Joint', 'Id', 'l_1', 'l_2', 'l_3', 'm_1', 'm_2', 'm_3', 'Value'])
            for i in trange(N):
                obj, camera_dist, target_height = self.sample_obj(obj_type, mean_flag, left_only, cute_flag=cute_flag)
                fname = os.path.join(self.save_dir, 'scene' + str(i).zfill(6) + '.xml')
                self.write_urdf(fname, obj.xml)
                self.scenes.append(fname)
                self.take_images(fname, obj, camera_dist, target_height, obj.joint_index, writ, i)

    def take_images(self, filename, obj, camera_dist, target_height, joint_index, writer, obj_no, n_frames=64):
        obj_id = pb.loadMJCF(filename)[0]
        base_pos, base_orn = pb.getBasePositionAndOrientation(obj_id)
        base_orn = np.roll(base_orn, 1)

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

        Ts = []
        for i in range(n_frames):
            str_id = f'{obj_no:02}_{i:04}'
            if self.mode == 1:
                if not i:
                    viewMatrix = pb.computeViewMatrix(
                        cameraEyePosition=[camera_dist, 0, target_height * 6],
                        cameraTargetPosition=[0, 0, target_height],
                        cameraUpVector=[0, 0, 1]
                    )
                    Ts.append(view2extrinsics(viewMatrix))
                pb.resetJointState(obj_id, joint_index, -2.3 * i / n_frames)
            else:
                if not i:
                    pb.resetJointState(obj_id, joint_index, np.random.uniform(-2.3, 0))
                    d_jit = np.random.uniform(-0.5, 0.5)
                    theta_s = np.random.uniform(-1 / 3, 1 / 3) * np.pi
                    eye_h_jit = np.random.uniform(-0.5, 0.5)
                    tgt_h_jit = np.random.uniform(-0.5, 0.5)
                d = camera_dist * (1 + d_jit - 0.5 * np.sign(d_jit) * i / n_frames)
                theta = theta_s - np.pi * 3 / 4 * i / n_frames * np.sign(theta_s)
                eye_h = target_height * 6 * (1 + eye_h_jit - 0.5 * np.sign(eye_h_jit) * i / n_frames)
                tgt_h = target_height * (1 + tgt_h_jit - 0.5 * np.sign(tgt_h_jit) * i / n_frames)
                viewMatrix = pb.computeViewMatrix(
                    cameraEyePosition=[d * np.cos(theta), d * np.sin(theta), eye_h],
                    cameraTargetPosition=[0, 0, tgt_h],
                    cameraUpVector=[0, 0, 1]
                )
                Ts.append(view2extrinsics(viewMatrix))
            width, height, img, depth, _ = pb.getCameraImage(
                calibrations.sim_width,
                calibrations.sim_height,
                viewMatrix,
                self.projectionMatrix,
                renderer=pb.ER_BULLET_HARDWARE_OPENGL
            )

            img = np.asarray(img).reshape((height, width, 4))
            depth = np.asarray(depth).reshape((height, width))
            real_depth = buffer_to_real(depth, self.d_far, self.d_near)
            norm_depth = real_depth / self.d_far

            if self.masked:
                # set background depth to 0
                mask = norm_depth < 0.99
                norm_depth *= mask

            if self.debugging:
                # save image to disk for visualization
                img_fname = os.path.join(self.save_dir, f'img{str_id}.png')
                depth_fname = os.path.join(self.save_dir, f'depth{str_id}.png')
                integer_depth = (norm_depth * 255).astype(np.uint8)
                Image.fromarray(img).save(img_fname)
                Image.fromarray(integer_depth).save(depth_fname)

            if joint_index is None:
                raise Exception("Joint index not defined! Are you simulating a 2DOF object? (Don't do that yet)")

            joint_info = pb.getJointInfo(obj_id, joint_index)
            pos = np.array(joint_info[14])    # joint position in parent frame
            l = np.array(joint_info[13])    # joint axis in local frame (ignored for JOINT_FIXED)
            orn = np.roll(joint_info[15], 1)  # joint orientation in parent frame (pybullet's quaternion is in x,y,z,w format); changes to w,x,y,z format, which is used by MJCF and transforms3d
            l = tf3d.quaternions.rotate_vector(l, orn)  # joint axis in base frame
            pos = tf3d.quaternions.rotate_vector(pos, base_orn) + base_pos
            l = tf3d.quaternions.rotate_vector(l, base_orn)
            m = np.cross(pos, l)
            joint_state = pb.getJointState(obj_id, joint_index)[0]
            row = np.concatenate((np.array([obj.name, obj.joint_type, str_id]), l, m, np.array([joint_state])))  # save joint's Plucker representation and state
            writer.writerow(row)

            depth_fname = os.path.join(self.save_dir, f'depth{str_id}.pt')
            torch.save(norm_depth, depth_fname)
        np.save(os.path.join(self.save_dir, 'cam_extrinsics{obj_no:02}.npy'), Ts)

        pb.removeBody(obj_id)
        if self.debugging:
            ffmpeg.input(f'{self.save_dir}/img{obj_no:02}_%04d.png').output(f'{self.save_dir}/vid{obj_no:02}.mp4').run()
