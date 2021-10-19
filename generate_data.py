import argparse
import os

from generation.inspect_data import make_animations


def main(args):
    # initialize generator
    if args.pybullet:
        from generation.generator_pybullet import SceneGenerator
    else:
        from generation.generator import SceneGenerator
    scene_gen = SceneGenerator(args.dir, args.mode, masked=args.masked, debug_flag=args.debug)

    if not args.eval_only:
        # generate train scenes
        scene_gen.generate_scenes(args.n, args.obj, mean_flag=args.mean, left_only=args.left_only, cute_flag=args.cute)

    # generate test scenes
    scene_gen.generate_scenes(args.n // 10, args.obj, test=True)

    # generate visualization for sanity
    if not args.pybullet and args.debug:
        make_animations(os.path.join(args.dir, args.obj), min(100, args.n * 16), use_color=args.debug)


parser = argparse.ArgumentParser(description="tool for generating articulated object data")
parser.add_argument('--n', type=int, default=1, help='number of examples to generate')
parser.add_argument('--dir', type=str, default='../microtrain/')
parser.add_argument('--obj', type=str, default='microwave')
parser.add_argument('--masked', action='store_true', default=False, help='remove background of depth images')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--mean', action='store_true', default=False, help='generate the mean object')
parser.add_argument('--cute', action='store_true', default=False, help='generate nice shots.')
parser.add_argument('--left-only', action='store_true', default=False, help='generate only left-opening cabinets')
parser.add_argument('--pybullet', action='store_true', default=False, help='render with PyBullet instead of MuJoCo')
parser.add_argument('--eval-only', action='store_true', default=False, help='only generate evaluation dataset')
parser.add_argument('--mode', type=int, choices=[1, 2], default=1, help='mode 1 generates articulated motion, mode 2 generates camera motion')
main(parser.parse_args())
