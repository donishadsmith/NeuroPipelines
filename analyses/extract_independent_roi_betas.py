"""
Extract average betas from independent (a-priori) sphere ROIs for each subject.

Notes
------
- Script is independent of all other scripts in pipeline other than ``first_level_glm.py``
or ``first_level_gPPI.py``
- Sphere masks are assumed to have been created from ``sphere_mask.py`` and are
assumed to have the following naming convention: "tpl-*_res-*_desc-sphere_mask_X_Y_Z.nii.gz"
"""

import argparse, sys
from pathlib import Path

import pandas as pd

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from bidsaid.cli import none_or

from _utils import (
    compute_average_betas,
    create_condition_label_str,
    delete_dir,
    exclude_beta_files,
    format_beta_df,
    get_beta_files,
    get_beta_names,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    get_individual_interpretations,
    get_qc_info,
    get_subject_beta_filenames,
    save_tabular_data,
    standardize_doses,
    validate_optional_path,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the average beta for each ROI at "
            "the individual level for downstream analysis."
        )
    )
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="Path to the BIDS root directory",
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Root of directory containing the first level beta coefficient images. "
            "Files are grabbed recursively."
        ),
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory.",
    )
    parser.add_argument(
        "--independent_roi_dir",
        dest="independent_roi_dir",
        required=True,
        help=(
            "Directory containing the sphere mask NIfTI files. "
            "Globs using '*_desc-sphere_mask_*.nii.gz'"
        ),
    )
    parser.add_argument(
        "--glm_dir",
        dest="glm_dir",
        required=False,
        default=None,
        type=none_or(),
        help=(
            "Used only when ``analysis_type`` is gPPI. Used to compute the "
            "individual average beta coefficient from the glm for the ROIs."
        ),
    )
    parser.add_argument(
        "--seed_mask_path",
        dest="seed_mask_path",
        required=False,
        default=None,
        type=none_or(),
        help=(
            "Path to the seed mask used as the seed for the gPPI. "
            "Used only when ``analysis_type`` is gPPI. "
            "Used to compute the average beta coefficient from the glm for the seed. "
            "This will only be used if ``glm_dir`` is not set to None."
        ),
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        help="Name of the task.",
    )
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        choices=["glm", "gPPI"],
        help="The type of analysis performed (glm or gPPI).",
    )
    parser.add_argument(
        "--exclude_niftis_file",
        dest="exclude_niftis_file",
        default=None,
        required=False,
        type=none_or(),
        help=(
            "Path to a file containing prefixes of the filename of the NIfTI images to exclude. "
            "Can list the fill name of the file (no parent directories) to exlude that specific file "
            "or can include the prefix (i.e., 'sub-101_task-nback_ses-01_space-MNI' or 'sub-101') to exclude all files starting "
            "with that prefix. Should contain a single column named 'nifti_prefix_filename' "
        ),
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


def build_data_table(bids_dir, subject_list, beta_files, cohort):
    """Lighter version of function in second_level.py"""
    bids_dir = Path(bids_dir)
    participants_df = pd.read_csv(bids_dir / "participants.tsv", sep="\t")

    session_files = sorted(list(bids_dir.rglob("sub-*_sessions.tsv")))
    sessions_dfs = []

    for session_file in session_files:
        sub_id = get_entity_value(session_file, "sub")
        if sub_id not in subject_list:
            continue

        df = pd.read_csv(session_file, sep="\t")
        df["participant_id"] = f"sub-{sub_id}"

        subject_beta_files = [str(f) for f in beta_files if sub_id in str(f)]
        for subject_beta_file in subject_beta_files:
            ses_id = get_entity_value(
                subject_beta_file, "ses", return_entity_prefix=True
            )
            df.loc[df["session_id"] == ses_id, "InputFile"] = subject_beta_file

            qc_info = get_qc_info(subject_beta_file)
            for key, value in qc_info.items():
                df.loc[df["session_id"] == ses_id, key] = value

        sessions_dfs.append(df)

    all_sessions = pd.concat(sessions_dfs, ignore_index=True)
    data_table = all_sessions.merge(participants_df, on="participant_id")
    data_table = data_table.dropna(how="all", axis=1)
    data_table = data_table.dropna(subset=["InputFile"])

    column_order = ["participant_id"] + [
        col for col in data_table.columns if col != "participant_id"
    ]
    data_table = data_table[column_order]

    return standardize_doses(data_table, cohort)


def get_sphere_masks(independent_roi_dir):
    roi_dir = Path(independent_roi_dir)
    masks = sorted(roi_dir.glob("*_desc-sphere_mask_*.nii.gz"))

    if not masks:
        LGR.warning(
            f"No sphere masks matching '*_desc-sphere_mask_*.nii.gz' "
            f"found in {roi_dir}"
        )

    return masks


def main(
    bids_dir,
    analysis_dir,
    dst_dir,
    independent_roi_dir,
    glm_dir,
    seed_mask_path,
    cohort,
    task,
    analysis_type,
    exclude_niftis_file,
    save_excel_version,
):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    glm_dir = validate_optional_path(glm_dir)
    seed_mask_path = validate_optional_path(seed_mask_path)

    sphere_masks = get_sphere_masks(independent_roi_dir)
    if not sphere_masks:
        LGR.info("No sphere masks found.")
        sys.exit(1)

    delete_dir(dst_dir / "independent_roi_betas")

    first_level_glt_labels = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="extract_independent_roi_betas"
    )

    for first_level_glt_label in first_level_glt_labels:
        LGR.info(f"TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_glt_label}")

        beta_files = get_beta_files(analysis_dir, task, first_level_glt_label)
        beta_files = exclude_beta_files(beta_files, exclude_niftis_file)

        if not beta_files:
            LGR.warning(f"No beta files found for {first_level_glt_label}")
            continue

        subject_list = sorted(set(get_entity_value(f.name, "sub") for f in beta_files))
        data_table = build_data_table(bids_dir, subject_list, beta_files, cohort)
        LGR.info(
            f"Built data table: {data_table.shape[0]} rows, "
            f"{len(subject_list)} subjects"
        )

        # The individual conditions for gPPI are main effects and should not be interpreted
        # since there is an interaction term in the model
        beta_names = get_beta_names(
            first_level_glt_label,
            create_sub_conditions=(analysis_type == "glm"),
        )

        for beta_name in beta_names:
            subject_beta_filenames = get_subject_beta_filenames(
                data_table,
                first_level_glt_label,
                beta_name,
            )
            if not subject_beta_filenames:
                LGR.warning(f"Skipping {beta_name}: no subject beta files.")
                continue

            for sphere_mask_filename in sphere_masks:
                beta_coefficient_df = data_table.copy(deep=True)

                LGR.info(
                    f"Extracting {analysis_type} betas for {beta_name} "
                    f"from {sphere_mask_filename.name}"
                )

                beta_coefficient_df[f"{analysis_type.upper()}_Individual_Roi_Beta"] = (
                    compute_average_betas(
                        beta_coefficient_df,
                        subject_beta_filenames,
                        sphere_mask_filename,
                        mask_origin="roi",
                        subject_col="participant_id",
                    )
                )

                beta_coefficient_df[
                    f"{analysis_type.upper()}_Individual_Beta_Interpretation"
                ] = get_individual_interpretations(
                    beta_coefficient_df,
                    beta_name,
                    mask_origin="roi",
                    analysis_type=analysis_type,
                )

                beta_coefficient_df["Condition_Label"] = create_condition_label_str(
                    beta_name
                )

                roi_coordinate = get_coordinate_from_filename(sphere_mask_filename)
                if roi_coordinate:
                    beta_coefficient_df["Roi_MNI_Coordinate"] = roi_coordinate

                if analysis_type == "gPPI" and glm_dir:
                    glm_beta_name = beta_name.replace("PPI_", "")

                    glm_subject_beta_filenames = get_subject_beta_filenames(
                        beta_coefficient_df,
                        first_level_glt_label,
                        glm_beta_name,
                        parent_path=glm_dir,
                    )

                    if not pd.Series(glm_subject_beta_filenames).isna().all():
                        beta_coefficient_df["GLM_Individual_Roi_Beta"] = (
                            compute_average_betas(
                                beta_coefficient_df,
                                glm_subject_beta_filenames,
                                sphere_mask_filename,
                                mask_origin="roi",
                                subject_col="participant_id",
                            )
                        )

                        beta_coefficient_df[
                            "GLM_Individual_Roi_Beta_Interpretation"
                        ] = get_individual_interpretations(
                            beta_coefficient_df,
                            beta_name,
                            mask_origin="roi",
                            analysis_type="glm",
                            remove_PPI_prefix=True,
                        )

                        if seed_mask_path:
                            possible_coordinate = get_coordinate_from_filename(
                                seed_mask_path
                            )
                            if possible_coordinate:
                                beta_coefficient_df["Seed_MNI_Coordinate"] = (
                                    possible_coordinate
                                )

                            LGR.info(
                                "Using the following seed mask path: "
                                f"{seed_mask_path}"
                            )

                            beta_coefficient_df["GLM_Individual_Seed_Beta"] = (
                                compute_average_betas(
                                    beta_coefficient_df,
                                    glm_subject_beta_filenames,
                                    seed_mask_path,
                                    mask_origin="seed",
                                    subject_col="participant_id",
                                )
                            )
                            beta_coefficient_df[
                                "GLM_Individual_Seed_Beta_Interpretation"
                            ] = get_individual_interpretations(
                                beta_coefficient_df,
                                beta_name,
                                mask_origin="seed",
                                analysis_type="glm",
                                remove_PPI_prefix=True,
                            )

                beta_coefficient_df = format_beta_df(analysis_type, beta_coefficient_df)

                add_condition_entity_key = beta_name != first_level_glt_label
                output_dir = (
                    dst_dir
                    / "independent_roi_betas"
                    / first_level_glt_label
                    / beta_name
                )
                output_csv_name = f"task-{task}_" + sphere_mask_filename.name.replace(
                    "-sphere_mask_", "_independent_roi_betas_"
                ).replace(".nii.gz", ".csv")
                save_tabular_data(
                    beta_coefficient_df,
                    output_dir,
                    output_csv_name,
                    first_level_glt_label,
                    beta_name,
                    add_condition_entity_key,
                    save_excel_version,
                )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
