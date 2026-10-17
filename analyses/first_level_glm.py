"""
Denoising papers:
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC9506314/
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC10619396/

- Aggressive denoising strategies can also remove task-signal, strategy should depend on
clinical population, type of analysis being done (activation vs connectivity, where in some cases
connectivity may require more aggressive denoising to ensure that correlation is not due to
shared nuisance variance), characteristics of data (numerous high-motion participants or
mostly low-motion participants), and whether strategies such as strict scrubbing (FD < 0.2) will
remove a significant amount of frames resulting in either suboptimal estimated beta coefficients
or too little retainerd participants. There is no optimal denoising strategy for all datasets.
"""

import argparse
from pathlib import Path

import pandas as pd
from bidsaid.logging import setup_logger
from bidsaid.metadata import get_tr, get_n_volumes
from bidsaid.cli import none_or, n_dummy_type, str_to_bool

from _denoising import (
    get_cosine_regressors,
    get_motion_regressors,
    percent_signal_change,
    perform_spatial_smoothing,
)
from _gen_afni_files import (
    create_censor_file,
    create_fd_file,
    create_binary_condition,
    create_timing_files,
    create_nuisance_regressor_file,
)
from _first_level_utils import (
    InvalidRunError,
    check_censoring,
    collect_acompcor_names,
    collect_session_files,
    create_diagnostic_condition_plots,
    filter_regressor_names,
    summarize_timing_conditions,
    validate_first_level_inputs,
)
from _models import (
    create_design_matrix,
    get_task_deconvolve_adults_cmd,
    get_task_deconvolve_kids_cmd,
    perform_first_level,
)
from _utils import (
    create_beta_files,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Perform first level GLM for a task.")
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="Path to BIDS directory."
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory.",
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of the derivatives directory.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=True,
        help="Template space.",
    )
    parser.add_argument(
        "--subject",
        dest="subject",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--filter_correct_trials",
        dest="filter_correct_trials",
        required=False,
        default=False,
        type=str_to_bool,
        help="Filter correct trials for event-related tasks.",
    )
    parser.add_argument(
        "--n_motion_parameters",
        dest="n_motion_parameters",
        default=6,
        type=int,
        choices=[6, 12, 18, 24],
        required=False,
        help=(
            "Number of motion parameters to use: 6 (base trans + rot), "
            "12 (base + derivatives), 18 (base + derivatives + power), "
            "24 (base + derivatives + power + derivative power)."
        ),
    )
    parser.add_argument(
        "--fd_threshold",
        dest="fd_threshold",
        default=0.5,
        type=float,
        required=False,
        help="Framewise displacement threshold.",
    )
    parser.add_argument(
        "--exclusion_criteria",
        dest="exclusion_criteria",
        default=0.30,
        type=float,
        required=False,
        help=(
            "Threshold to exclude entire run if more than the specified "
            "percentage of steady-state volumes are removed."
        ),
    )
    parser.add_argument(
        "--n_dummy_scans",
        dest="n_dummy_scans",
        default="auto",
        type=n_dummy_type,
        required=False,
        help=(
            "Number of dummy scans to remove. If 'auto' computes number of dummy scans "
            "by the numnber of 'non_steady_state_outlier_XX' columns."
        ),
    )
    parser.add_argument(
        "--n_acompcor",
        dest="n_acompcor",
        default=5,
        type=int,
        required=False,
        help="Number of aCompCor components.",
    )
    parser.add_argument(
        "--acompcor_strategy",
        dest="acompcor_strategy",
        default="separate",
        choices=["combined", "separate", "none"],
        required=False,
        help="Whether to use 'combined' aCompCor, 'separate' components, or 'none'",
    )
    parser.add_argument(
        "--fwhm",
        dest="fwhm",
        default=6,
        type=int,
        required=False,
        help="Spatial blurring.",
    )
    parser.add_argument(
        "--exclude_niftis_file",
        dest="exclude_niftis_file",
        default=None,
        required=False,
        type=none_or(),
        help=(
            "File containing prefixes of the filename of the NIfTI images to exclude. "
            "Can list the fill name of the file (no parent directories) to exlude that specific file "
            "or can include the prefix (i.e., 'sub-101_task-nback_ses-01_space-MNI' or 'sub-101') to exclude all files starting "
            "with that prefix. Should contain a single column named 'nifti_prefix_filename' "
        ),
    )

    return parser


def main(
    bids_dir,
    afni_img_path,
    dst_dir,
    deriv_dir,
    space,
    cohort,
    subject,
    task,
    filter_correct_trials,
    n_motion_parameters,
    fd_threshold,
    exclusion_criteria,
    n_dummy_scans,
    n_acompcor,
    acompcor_strategy,
    fwhm,
    exclude_niftis_file,
):
    report_dir, layout, sessions = validate_first_level_inputs(
        dst_dir,
        bids_dir,
        deriv_dir,
        cohort,
        task,
        subject,
        analysis_type="glm",
    )

    for session in sessions:
        report, report_path, session_files, skip_iteration = collect_session_files(
            report_dir,
            layout,
            subject,
            session,
            task,
            space,
            acompcor_strategy,
            exclude_niftis_file,
            analysis_type="glm",
        )
        if skip_iteration:
            continue

        confounds_tsv_file = session_files.confounds_tsv_file
        confounds_json_file = session_files.confounds_json_file
        event_file = session_files.event_file
        mask_file = session_files.mask_file
        nifti_file = session_files.nifti_file

        subject_dir = (
            Path(dst_dir) / f"sub-{subject}" / f"ses-{session}" / "func" / task
        )
        subject_dir.mkdir(parents=True, exist_ok=True)

        confounds_df = pd.read_csv(confounds_tsv_file, sep="\t").fillna(0)

        try:
            censor_mask, censor_info = check_censoring(
                subject,
                session,
                task,
                confounds_df,
                n_dummy_scans,
                fd_threshold,
                exclusion_criteria,
            )
        except InvalidRunError as exc:
            report.mark_excluded(str(exc))
            report.create_report(report_path, "first_level.html")
            continue

        report.add_context(
            fd_threshold=fd_threshold,
            exclusion_criteria=exclusion_criteria,
            n_censored_volumes=censor_info.n_censored,
            n_total_volumes=censor_info.n_total,
            percent_censored=censor_info.percent_censored,
            dummy_method=censor_info.dummy_method,
            n_dummy_scans=censor_info.n_dummy_scans,
            mean_fd_before_censoring=censor_info.mean_fd_before_censoring,
            mean_fd_after_censoring=censor_info.mean_fd_after_censoring,
        )

        censor_file = create_censor_file(
            subject_dir, subject, session, task, space, censor_mask
        )

        high_motion_only_mask = censor_mask.copy()
        high_motion_only_mask[: censor_info.n_dummy_scans] = 1
        create_censor_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            high_motion_only_mask,
            desc="high_motion_outliers_only",
        )

        create_fd_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            confounds_df,
            censor_mask,
            censor_info.n_dummy_scans,
        )

        cosine_regressors, cosine_regressor_names = get_cosine_regressors(confounds_df)

        motion_regressors, motion_regressor_names = get_motion_regressors(
            confounds_df,
            n_motion_parameters,
        )

        acompcor_regressors, acompcor_regressor_names = collect_acompcor_names(
            confounds_json_file, confounds_df, acompcor_strategy, n_acompcor
        )

        report.add_context(
            n_motion_parameters=n_motion_parameters,
            motion_regressor_names=motion_regressor_names,
            acompcor_strategy=acompcor_strategy,
            n_acompcor=n_acompcor,
            acompcor_component_names=acompcor_regressor_names or [],
            cosine_parameter_names=cosine_regressor_names,
            fwhm=fwhm,
            filter_correct_trials=filter_correct_trials,
        )

        regressor_names = filter_regressor_names(
            cosine_regressor_names, motion_regressor_names, acompcor_regressor_names
        )
        nuisance_regressors_file, report_info = create_nuisance_regressor_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            censor_mask,
            regressor_names,
            cosine_regressors,
            motion_regressors,
            acompcor_regressors,
        )

        report.add_context(
            dropped_regressors=(
                report_info.collinear_regressor_names
                + report_info.constant_column_names
            )
        )

        timing_dir = create_timing_files(
            subject_dir,
            event_file,
            task,
            filter_correct_trials,
            append_task_name=False,
        )

        tr = get_tr(nifti_file)
        n_volumes = get_n_volumes(nifti_file)

        condition_filenames_dict = create_binary_condition(
            afni_img_path,
            timing_dir,
            cohort,
            task,
            tr,
            n_volumes,
            censor_file,
        )

        diagnostic_condition_plots = create_diagnostic_condition_plots(
            condition_filenames_dict,
            tr,
            fd_threshold,
        )
        report.add_context(diagnostic_condition_plots=diagnostic_condition_plots)

        timing_conditions, event_type = summarize_timing_conditions(timing_dir, task)
        report.add_context(timing_conditions=timing_conditions, event_type=event_type)

        percent_change_nifti_file = percent_signal_change(
            subject_dir,
            afni_img_path,
            nifti_file,
            mask_file,
            censor_file,
        )

        smoothed_nifti_file = perform_spatial_smoothing(
            subject_dir,
            afni_img_path,
            percent_change_nifti_file,
            mask_file,
            fwhm,
        )

        get_task_deconvolve_cmd = {
            "kids": get_task_deconvolve_kids_cmd,
            "adults": get_task_deconvolve_adults_cmd,
        }

        deconvolve_cmd = get_task_deconvolve_cmd[cohort](
            task,
            timing_dir,
            nuisance_regressors_file,
            analysis_type="glm",
        )

        report.add_context(
            deconvolve_cmd=f"{deconvolve_cmd['num_stimts']} {deconvolve_cmd['args']}".replace(
                "  ", " "
            ),
        )

        design_matrix_file = create_design_matrix(
            subject_dir,
            afni_img_path,
            smoothed_nifti_file,
            mask_file,
            censor_file,
            deconvolve_cmd,
            cosine_regressor_names,
        )

        stats_file_relm = perform_first_level(
            subject_dir,
            afni_img_path,
            design_matrix_file,
            smoothed_nifti_file,
            mask_file,
        )

        betas_dir = stats_file_relm.parent / "betas"
        betas_dir.mkdir(parents=True, exist_ok=True)

        create_beta_files(
            stats_file_relm,
            betas_dir,
            afni_img_path,
            cohort,
            task,
            analysis_type="glm",
        )

        report.create_report(report_path, "first_level.html")


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
