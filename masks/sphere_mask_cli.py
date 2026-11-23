import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sphere_mask import run_pipeline

from _general_utils import _convert_to_bool


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for generating a-priori sphere masks for seed-based connectivity analyses."
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["kids", "adults"],
        help="The name of the cohort.",
    )
    parser.add_argument(
        "--coordinate",
        dest="coordinate",
        required=True,
        nargs=3,
        help="The coordinate in MNI or Talairach space.",
    )
    parser.add_argument(
        "--sphere_radius",
        dest="sphere_radius",
        required=True,
        type=float,
        help="The radius of the sphere mask in mm.",
    )
    parser.add_argument(
        "--original_coordinate_space",
        dest="original_coordinate_space",
        required=False,
        default="MNI",
        choices=["MNI", "Talairach"],
        type=str,
        help="The original space of the coordinate.",
    )
    parser.add_argument(
        "--transform_method",
        dest="transform_method",
        required=False,
        default="Lancaster",
        choices=["Lancaster", "Brett"],
        type=str,
        help=(
            "Method for transforming the Talairach coordinate to MNI space. "
            "Use the Lancaster method unless the paper is pre-2007/2008 or specifically "
            "states that the Brett transform was used."
        ),
    )
    parser.add_argument(
        "--use_black_bg",
        dest="dst_dir",
        required=False,
        default=False,
        type=_convert_to_bool,
        help="Whether or not to use a black background in an image.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="sphere_dst_dir",
        required=False,
        default=None,
        help=(
            "The root of the output directory for the sphere mask and plot. "
            "Nothing is saved if no location is given."
        ),
    )

    return parser


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    sphere_filename, plot_filename, _, _ = run_pipeline(**vars(args))

    if args.sphere_dst_dir:
        print(f"Sphere NIfTI saved to: {sphere_filename}")
        print(f"Plot saved to: {plot_filename}")
