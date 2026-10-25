"""Standolone file for thresholding Palm outputs"""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bidsaid.logging import setup_logger

from _second_level_utils import threshold_palm_output
from _utils import (
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Threshold images.")
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help="Root path to directory containing the PALM permutation files.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination directory for analysis.",
    )
    parser.add_argument(
        "--cluster_correction_p",
        dest="cluster_correction_p",
        default=0.05,
        type=float,
        required=False,
        help=(
            "Significance threshold for cluster significance for nonparametric. "
            "This script uses the threshold free cluster enhancement approach which "
            "eliminates the need to select an arbritrary threshold for the voxels (cluster-forming threshold)"
        ),
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        choices=["glm", "gPPI"],
        help="The type of analysis performed (glm or gPPI).",
    )

    return parser


def get_output_prefixes(
    analysis_dir, task, first_level_gltlabel, second_level_glt_code
):
    entity_key = get_contrast_entity_key(first_level_gltlabel)
    prefix = f"task-{task}_{entity_key}-{first_level_gltlabel}_gltcode-{second_level_glt_code}_desc-nonparametric"
    files = list(analysis_dir.rglob(f"*{prefix}*"))

    return files[0].parent / prefix


def main(analysis_dir, dst_dir, cohort, task, analysis_type, cluster_correction_p):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}")

    first_level_gltlabels = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="threshold_palm_imgs"
    )
    for first_level_gltlabel in first_level_gltlabels:
        for second_level_glt_code in get_second_level_glt_codes(cohort):
            output_prefix = get_output_prefixes(
                analysis_dir, task, first_level_gltlabel
            )
            threshold_palm_output(
                output_prefix, second_level_glt_code, cluster_correction_p
            )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
