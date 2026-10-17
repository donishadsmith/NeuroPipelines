import argparse, itertools, subprocess
from pathlib import Path

import pandas as pd

from bidsaid.logging import setup_logger

from _utils import (
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description=("Use AFNI's whereami and identify significant clusters.")
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Root path to directory containing the cluster table files "
            "outputted by Nilearn's get_cluster_table"
        ),
    )
    parser.add_argument(
        "--scratch_dir",
        dest="scratch_dir",
        required=True,
        help="Path to the scratch directory.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--method",
        dest="method",
        required=False,
        default="parametric",
        choices=["parametric", "nonparametric"],
        help="Whether parametric (3dlmer) or nonparametric (Palm) was used.",
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
    parser.add_argument(
        "--atlas",
        dest="atlas",
        required=True,
        help="The atlas to use.",
    )
    parser.add_argument(
        "--save_excel_version",
        dest="save_excel_version",
        required=False,
        default=False,
        help=(
            "Save Excel version of the cluster tables "
            "to allow for certain Excel features such as highlighting"
        ),
    )

    return parser


def identify_mni_regions(afni_img_path, coord_filename, atlas):
    cmd = (
        f"apptainer exec -B /scratch:/scratch -B /projects:/projects {afni_img_path} whereami "
        f"-lpi -coord_file {coord_filename} -space MNI_2009c_asym -atlas {atlas}"
    )
    std_output = subprocess.run(
        cmd, shell=True, check=True, capture_output=True, text=True
    ).stdout.splitlines()

    header_sentence = "Original input data coordinates"
    target_line_headers = ["Focus point:", "* Within", "Within"]

    data = {"Region": [], "Label Spatial Precision": []}

    lines = [line.strip() for line in std_output if line]
    block_indices = [
        index for index, line in enumerate(lines) if line.startswith(header_sentence)
    ]
    for pos, start_index in enumerate(block_indices):
        identified_region = False
        start_index = block_indices[pos]
        stop_index = (
            block_indices[pos + 1] if pos + 1 != len(block_indices) else len(lines)
        )
        for line in lines[start_index:stop_index]:
            if any(
                line.startswith(target_line_header)
                for target_line_header in target_line_headers
            ):
                identified_region = True
                split_line = line.split(":")
                region_name = split_line[-1].strip()
                header_name = split_line[0].removeprefix("*").strip()

                break

        if not identified_region:
            region_name = float("NaN")
            header_name = float("NaN")

        data["Region"].append(region_name)
        data["Label Spatial Precision"].append(header_name)

    return data


def add_region_information_to_data(
    cluster_table_filename,
    scratch_dir,
    afni_img_path,
    task,
    atlas,
    save_excel_version,
):
    coord_filename = (
        scratch_dir / task / cluster_table_filename.name.replace(".csv", ".1D")
    )
    coord_filename.parent.mkdir(parents=True, exist_ok=True)

    cluster_table = pd.read_csv(cluster_table_filename, sep=None, engine="python")
    cluster_table[["X", "Y", "Z"]].astype(str).to_csv(
        coord_filename,
        sep=" ",
        header=False,
        index=False,
    )

    data = identify_mni_regions(afni_img_path, coord_filename, atlas)

    # Delete columns if exist in data
    for col in data:
        if col in cluster_table.columns:
            cluster_table = cluster_table.drop(col, axis=1)

    cluster_table = pd.concat([cluster_table, pd.DataFrame(data)], axis=1)

    cluster_table.to_csv(cluster_table_filename, sep=",", index=False)

    if save_excel_version:
        cluster_table.to_excel(
            str(cluster_table_filename).replace(".csv", ".xlsx"), index=False
        )


def main(
    analysis_dir,
    scratch_dir,
    afni_img_path,
    method,
    cohort,
    task,
    analysis_type,
    atlas,
    save_excel_version,
):
    analysis_dir = Path(analysis_dir)
    scratch_dir = Path(scratch_dir)

    LGR.info(f"TASK: {task}")

    first_level_glt_label = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="identify_cluster_regions"
    )
    first_level_glt_label_list = list(
        itertools.product(
            first_level_glt_label,
            get_second_level_glt_codes(cohort),
        )
    )
    for first_level_glt_label, second_level_glt_code in first_level_glt_label_list:
        entity_key = get_contrast_entity_key(first_level_glt_label)
        cluster_table_filename = list(
            analysis_dir.rglob(
                f"task-{task}_{entity_key}-{first_level_glt_label}"
                f"_gltcode-{second_level_glt_code}_desc-{method}_cluster_results.csv"
            )
        )

        if not cluster_table_filename:
            continue

        LGR.info(
            f"FIRST LEVEL GLTLABEL: {first_level_glt_label} "
            f"SECOND LEVEL GLTCODE: {second_level_glt_code}"
        )

        cluster_table_filename = cluster_table_filename[0]
        add_region_information_to_data(
            cluster_table_filename,
            scratch_dir,
            afni_img_path,
            task,
            atlas,
            save_excel_version,
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
