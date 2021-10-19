import numpy as np
import pyro
import pyro.distributions as dist

from generation.ArticulatedObjs import ArticulatedObject
from generation.utils import (angle_to_quat, make_quat_string, make_string,
                              sample_pose)

d_len = dist.Uniform(10 / 2 * 0.0254, 22 / 2 * 0.0254)
d_width = dist.Uniform(16 / 2 * 0.0254, 30 / 2 * 0.0254)
d_height = dist.Uniform(9 / 2 * 0.0254, 18 / 2 * 0.0254)
d_thic = dist.Uniform(0.01 / 2, 0.03 / 2)


def sample_microwave(mean_flag):
    if mean_flag:
        print('Using mean microwave')
        length = d_len.mean
        width = d_width.mean
        height = d_height.mean
        thickness = d_thic.mean
    else:
        length = pyro.sample('length', d_len).item()
        width = pyro.sample('width', d_width).item()
        height = pyro.sample('height', d_height).item()
        thickness = pyro.sample('thic', d_thic).item()
    left = True
    mass = pyro.sample('mass', dist.Uniform(5.0, 30.0)).item()
    return length, width, height, thickness, left, mass


def sample_handle(height):
    HANDLE_LEN = pyro.sample('hl', dist.Uniform(0.01, 0.03)).item()
    HANDLE_WIDTH = pyro.sample('hw', dist.Uniform(0.01, 0.03)).item()
    HANDLE_HEIGHT = pyro.sample('hh', dist.Uniform(0.1, height)).item()
    return HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT


def const_handle(height):
    HANDLE_LEN = 0.01
    HANDLE_WIDTH = 0.01
    HANDLE_HEIGHT = height
    return HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT


def build_microwave(length, width, height, thic, left, set_pos=None, set_rot=None):
    if set_pos is None:
        base_xyz, base_angle = sample_pose()
        base_quat = angle_to_quat(base_angle)
    else:
        base_xyz = set_pos
        base_quat = set_rot

    base_origin = make_string(tuple(base_xyz))
    base_orientation = make_quat_string(tuple(base_quat))

    base_size = make_string((length, width, thic))
    side_size = make_string((length, thic, height))

    back_size = make_string((thic, width, height))
    top_size = base_size
    door_size = make_string((thic, width * 0.75, height))

    left_origin = make_string((0, -width + thic, height))
    right_origin = make_string((0, width - thic, height))
    top_origin = make_string((0, 0, height * 2))
    back_origin = make_string((-length + thic, 0.0, height))

    keypad_size = make_string((length - thic, width * 0.25 - thic, height - thic))
    keypad_origin = make_string((thic, width * 0.75 - thic, height))

    hinge_origin = make_string((length, -width, height))
    hinge_range = '"-2.3 0"'
    door_origin = make_string((0.0, width * 0.75, 0.0))

    HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT = const_handle(height)
    handle_origin = make_string((HANDLE_LEN + thic, width * 1.25, 0))
    handle_size = make_string((HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT))

    geometry = np.array([length, width, height, left])  # length = 4
    parameters = np.array([[length, -width, height], [0.0, width, 0.0]])  # shape = 2*3, length = 6
    cab = ArticulatedObject(0, geometry, parameters, '', base_xyz, base_quat)

    xml = write_xml(base_origin, base_orientation, base_size, left_origin,
                    right_origin, side_size, top_origin, top_size, back_origin,
                    back_size, keypad_origin, keypad_size, hinge_origin, hinge_range,
                    door_origin, door_size, handle_origin, handle_size)

    cab.joint_index = 5
    cab.name = 'microwave'
    cab.joint_type = 'revolute'
    cab.xml = xml
    return cab


def write_xml(base_origin, base_orientation, base_size, left_origin,
              right_origin, side_size, top_origin, top_size, back_origin,
              back_size, keypad_origin, keypad_size, hinge_origin, hinge_range,
              door_origin, door_size, handle_origin, handle_size):
    return '''
<mujoco model="Microwave Oven">
    <compiler angle="radian"/>
    <worldbody>
        <body name="bottom" pos=''' + base_origin + ''' quat=''' + base_orientation + '''>
            <inertial pos="0 0 0" mass="0" diaginertia="1 1 1"/>
            <geom size=''' + base_size + ''' type="box"/>
            <body name="left" pos=''' + left_origin + '''>
                <geom size=''' + side_size + ''' type="box"/>
            </body>
            <body name="right" pos=''' + right_origin + '''>
                <geom size=''' + side_size + ''' type="box"/>
            </body>
            <body name="top" pos=''' + top_origin + '''>
                <geom size=''' + top_size + ''' type="box"/>
            </body>
            <body name="back" pos=''' + back_origin + ''' >
                <geom size=''' + back_size + ''' type="box"/>
            </body>
            <body name="keypad" pos=''' + keypad_origin + '''>
                <geom size=''' + keypad_size + ''' type="box"/>
            </body>
            <body name="hinge" pos=''' + hinge_origin + '''>
                <joint type="hinge" pos="0 0 0" axis="0 0 1" limited="true" range=''' + hinge_range + '''/>
                <body name="door" pos=''' + door_origin + '''>
                    <geom size=''' + door_size + ''' type="box"/>
                </body>
                <body name="handle" pos=''' + handle_origin + '''>
                    <geom size=''' + handle_size + ''' type="box"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>'''


def test():
    l, w, h, t, left, m = sample_microwave(False)
    cab = build_microwave(l, w, h, t, left, set_pos=[0.9, 0.0, -0.15], set_rot=[0, 0, 0, 1])
    model = load_model_from_xml(cab.xml)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    t = 0
    sim.data.ctrl[0] = - 0.2
    while t < 5000:
        sim.step()
        viewer.render()
        t += 1


if __name__ == "__main__":
    from mujoco_py import MjSim, MjViewer, load_model_from_xml
    for i in range(200):
        test()
